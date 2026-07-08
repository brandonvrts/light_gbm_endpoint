"""Train the brokerage conversion model on the CORRECTED, time-localized label
(replaces the leaky rea.ProspectLabelSnapshot stand-in) using only features that
exist in the current window. Scoop/intent features are dropped (zi.Scoop has no
history before 2026-03), pending a ZoomInfo backfill.

Label (read-only): a prospect converts if its tenant (rea.vw_Deal.tenantId ->
dbo.CommAmendmentSnapshot.hTenant) has a signed New/Renewal/Expansion/Relocation
amendment (iType 0/1/2/15) with COALESCE(dtSign,dtStart) within 180d AFTER the
ProspectFeatureSnapshot date. Group-disjoint (by hMy) pooled CV; data-limited.

Run: PYTHONPATH=. .venv/bin/python scripts/train_brokerage_corrected.py
NEVER writes to Yardi/InsightMart (SELECT-only); writes only the local bundle.
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             brier_score_loss, log_loss)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core import db                                                      # noqa: E402
from contracts.signal_desk_brokerage import prepare_features as bk_prep  # noqa: E402
from contracts.signal_desk_brokerage.schema import FEATURES              # noqa: E402

BUNDLES = ROOT / "bundles"
TS = datetime.now().strftime("%Y%m%d_%H%M%S")
VERSION = "v0.2-corrected-scoopfree"

# Scoop/intent family — zi.Scoop only exists since 2026-03, so these are ~0 across
# most of the 2024-11..2026-04 window. Dropped until a dated ZoomInfo backfill exists.
SCOOP = ["scoop_count_90d", "scoop_count_365d", "days_since_last_scoop",
         "had_funding_scoop_180d", "had_leadership_move_180d", "had_office_relo_365d",
         "had_hiring_scoop_365d", "had_layoff_scoop_180d"]
KEEP = [f for f in FEATURES if f not in SCOOP]   # 29 scoop-free features


def load_data() -> pd.DataFrame:
    cols = ", ".join(f"f.[{c}]" for c in KEEP)
    feat = db.query(
        f"SELECT f.hMy, f.SnapshotDate, {cols} FROM [{db.DB_MAIN}].rea.ProspectFeatureSnapshot f")
    conv = db.query(
        f"SELECT DISTINCT d.commProspectId AS hMy, COALESCE(a.dtSign, a.dtStart) AS conv_date "
        f"FROM [{db.DB_MAIN}].rea.vw_Deal d "
        f"JOIN [{db.DB_MAIN}].dbo.CommAmendmentSnapshot a ON a.hTenant = d.tenantId "
        f"WHERE a.iType IN (0,1,2,15) AND COALESCE(a.dtSign, a.dtStart) IS NOT NULL")
    feat["hMy"] = feat["hMy"].astype("int64")
    feat["SnapshotDate"] = pd.to_datetime(feat["SnapshotDate"])
    conv["hMy"] = conv["hMy"].astype("int64")
    conv["conv_date"] = pd.to_datetime(conv["conv_date"])
    cmap: dict[int, list] = defaultdict(list)
    for r in conv.itertuples(index=False):
        cmap[int(r.hMy)].append(r.conv_date)

    def label(hmy: int, sd) -> int:
        ds = cmap.get(int(hmy))
        if not ds:
            return 0
        hi = sd + pd.Timedelta(days=180)
        return int(any((d > sd) and (d <= hi) for d in ds))

    feat["Label"] = [label(h, s) for h, s in zip(feat["hMy"], feat["SnapshotDate"])]
    return feat


def _model():
    return LGBMClassifier(n_estimators=150, num_leaves=15, max_depth=4, learning_rate=0.03,
                          min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
                          reg_lambda=5.0, is_unbalance=True, n_jobs=4, verbose=-1, random_state=7)


def main():
    print("Loading features + building corrected bridge label (read-only)...")
    df = load_data()
    y = df["Label"].to_numpy().astype(int)
    groups = df["hMy"].to_numpy()
    medians = bk_prep.fit_prep(df)           # fit_prep guards missing cols -> 0.0
    X = pd.DataFrame(index=df.index)         # build only the 29 KEEP cols (scoops weren't pulled)
    for c in KEEP:
        X[c] = pd.to_numeric(df[c], errors="coerce").fillna(medians.get(c, 0.0)).astype(float)

    n_rows, n_pos = len(y), int(y.sum())
    n_groups = df["hMy"].nunique()
    n_pos_groups = df.loc[y == 1, "hMy"].nunique()
    base = float(y.mean())
    print(f"  rows={n_rows}  positives={n_pos} ({base:.3%})  prospects={n_groups}  "
          f"converting_prospects={n_pos_groups}  features={len(KEEP)}")
    if n_pos_groups < 2 or len(set(y)) < 2:
        print("  Not enough positives to train. Abort."); return

    # Group-disjoint pooled out-of-fold metrics (a prospect never spans train/test).
    n_splits = min(5, n_pos_groups)
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.full(n_rows, np.nan)
    for tr, te in gkf.split(X, y, groups):
        m = _model(); m.fit(X.iloc[tr], y[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
    mask = ~np.isnan(oof)
    met = {
        "cv": f"group-disjoint {n_splits}-fold (by hMy), pooled OOF",
        "pr_auc": round(float(average_precision_score(y[mask], oof[mask])), 4),
        "pr_auc_no_skill": round(base, 4),
        "roc_auc": round(float(roc_auc_score(y[mask], oof[mask])), 4),
        "brier": round(float(brier_score_loss(y[mask], oof[mask])), 4),
        "log_loss": round(float(log_loss(y[mask], oof[mask], labels=[0, 1])), 4),
        "n_rows": n_rows, "n_pos_rows": n_pos, "base_rate": round(base, 4),
        "n_prospects": int(n_groups), "n_converting_prospects": int(n_pos_groups),
    }
    print("  CV metrics:", json.dumps(met))

    # Final served model: calibrated on all data (fallback to plain if calib infeasible).
    try:
        model = CalibratedClassifierCV(_model(), method="sigmoid", cv=3); model.fit(X, y)
        calib = "sigmoid"
    except Exception as e:
        print("  calibration fallback:", str(e)[:120]); model = _model(); model.fit(X, y); calib = "none"

    bundle = {
        "model": model, "feature_columns": KEEP, "model_input_features": KEEP,
        "feature_medians": medians, "bundle_format": "corrected-scoopfree",
        "thresholds": {"risk_low_max": 0.33, "risk_medium_max": 0.66},
        # floor below the calibrated median (~0.003) so the rare-event spread isn't flattened
        "calibration": {"method": calib, "prob_floor": 0.002, "prob_ceil": 0.99},
        "metrics": met, "version": VERSION, "trained_at": TS, "algorithm": "lgbm",
        "horizon_days": 180,
        "label": "corrected bridge: vw_Deal.tenantId->CommAmendmentSnapshot.hTenant, "
                 "iType 0/1/2/15, COALESCE(dtSign,dtStart) within 180d after snapshot",
        "source": "real-corrected-label",
        "dropped_features": SCOOP,
    }
    manifest = {
        "contract": "signal-desk-brokerage", "version": VERSION, "feature_count": len(KEEP),
        "source": "real-corrected-label", "n_train": n_rows, "n_pos_rows": n_pos,
        "n_converting_prospects": int(n_pos_groups), "trained_at": TS, "metrics": met,
        "note": "Corrected, time-localized label via the deal->amendment bridge (replaces the "
                "leaky ProspectLabelSnapshot). Scoop-free (zi.Scoop has no pre-2026-03 history). "
                "DATA-LIMITED / INTERIM: few converting prospects in the 2024-11..2026-04 feature "
                "window; not yet a production-validated model. Pending historical feature backfill "
                "+ ZoomInfo scoop history to reach ~500 positives.",
    }
    d = BUNDLES / "signal_desk_brokerage" / VERSION
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, d / "bundle.joblib")
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    (BUNDLES / "signal_desk_brokerage" / "default.txt").write_text(VERSION)
    print(f"  saved {VERSION} (calibration={calib}) and pointed default.txt -> {VERSION}")
    print("DONE")


if __name__ == "__main__":
    main()
