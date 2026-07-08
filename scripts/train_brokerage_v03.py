"""Brokerage conversion model v0.3 — small-n-defensible stack.

Corrected bridge label (same as v0.2) on the 2024-11..2026-04 window, 29 scoop-free
features. Then:
  1. elastic-net logistic selects ~<=8 features (Riley parameter budget for ~54 events),
  2. fit BOTH a penalized (ridge) logistic and a Firth+FLIC logistic on the selected feats
     (NO class weighting / NO resampling),
  3. evaluate with expanding-window rolling-origin walk-forward (train past quarters, test
     next) + paired-bootstrap 95% CI of PR-AUC lift over the no-skill base rate,
  4. promote the cleaner model; calibrate the served penalized model with Platt (Firth uses
     its FLIC intercept correction). Threshold-shift, not resampling.

Run: PYTHONPATH=. .venv/bin/python scripts/train_brokerage_v03.py
SELECT-only DB reads; writes only the local bundle.
"""
from __future__ import annotations
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, log_loss

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core import db                                                      # noqa: E402
from contracts.signal_desk_brokerage import prepare_features as bk_prep  # noqa: E402
from contracts.signal_desk_brokerage.schema import FEATURES              # noqa: E402
from firth_logistic import FirthLogisticRegression                       # noqa: E402

BUNDLES = ROOT / "bundles"
TS = datetime.now().strftime("%Y%m%d_%H%M%S")
VERSION = os.environ.get("GBME_BUNDLE_VERSION", "v0.3-corrected-logistic")
SCOOP = ["scoop_count_90d", "scoop_count_365d", "days_since_last_scoop",
         "had_funding_scoop_180d", "had_leadership_move_180d", "had_office_relo_365d",
         "had_hiring_scoop_365d", "had_layoff_scoop_180d"]
KEEP = [f for f in FEATURES if f not in SCOOP]
RNG = np.random.default_rng(7)


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
    cmap = defaultdict(list)
    for r in conv.itertuples(index=False):
        cmap[int(r.hMy)].append(r.conv_date)

    def label(hmy, sd):
        ds = cmap.get(int(hmy))
        if not ds:
            return 0
        hi = sd + pd.Timedelta(days=180)
        return int(any((d > sd) and (d <= hi) for d in ds))

    feat["Label"] = [label(h, s) for h, s in zip(feat["hMy"], feat["SnapshotDate"])]
    feat["q"] = feat["SnapshotDate"].dt.year * 4 + (feat["SnapshotDate"].dt.month - 1) // 3
    return feat


def matrix(df, feats):
    med = bk_prep.fit_prep(df)
    X = pd.DataFrame(index=df.index)
    for c in feats:
        X[c] = pd.to_numeric(df[c], errors="coerce").fillna(med.get(c, 0.0)).astype(float)
    return X, med


def select_features(df, y):
    X29, _ = matrix(df, KEEP)
    Z = StandardScaler().fit_transform(X29.values)
    enet = LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5,
                              C=0.2, max_iter=5000, random_state=7)
    enet.fit(Z, y)
    coef = np.abs(enet.coef_[0])
    order = [i for i in np.argsort(coef)[::-1] if coef[i] > 1e-8][:8]
    return [KEEP[i] for i in order]


def _penalized():
    return Pipeline([("sc", StandardScaler()),
                     ("lr", LogisticRegression(penalty="l2", C=1.0, max_iter=2000, random_state=7))])


def cal_intercept_slope(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    lo = np.log(p / (1 - p))
    lr = LogisticRegression(penalty=None, max_iter=1000).fit(lo.reshape(-1, 1), y)
    return float(lr.intercept_[0]), float(lr.coef_[0][0])   # ideal: 0, 1


def rolling_origin(df, X, y, build):
    """Expanding-window: train quarters < q, test quarter == q. Start testing once
    >=10 positive prospects are in the training history. Returns pooled OOF + per-fold."""
    qs = sorted(df["q"].unique())
    oof_idx, oof_p = [], []
    per_fold = []
    for q in qs:
        tr = df["q"] < q
        te = df["q"] == q
        if te.sum() == 0:
            continue
        tr_pos_groups = df.loc[tr & (y == 1), "hMy"].nunique()
        if tr_pos_groups < 10:
            continue
        m = build(); m.fit(X[tr].values, y[tr])
        p = m.predict_proba(X[te].values)[:, 1]
        idx = np.where(te.values)[0]
        oof_idx.extend(idx.tolist()); oof_p.extend(p.tolist())
        yte = y[te]
        if yte.sum() >= 1:
            per_fold.append({"q": int(q), "n": int(te.sum()), "pos": int(yte.sum()),
                             "pr_auc": round(float(average_precision_score(yte, p)), 4) if yte.sum() >= 1 else None})
    oof_idx = np.array(oof_idx); oof_p = np.array(oof_p)
    yo = y[oof_idx]
    return yo, oof_p, per_fold


def paired_bootstrap_lift(y, p, n=2000):
    base = float(y.mean())
    diffs = []
    for _ in range(n):
        idx = RNG.integers(0, len(y), len(y))
        if y[idx].sum() < 2:
            continue
        diffs.append(average_precision_score(y[idx], p[idx]) - y[idx].mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return round(base, 4), round(float(np.mean(diffs)), 4), round(float(lo), 4), round(float(hi), 4)


def evaluate(name, df, X, y, build):
    yo, po, per_fold = rolling_origin(df, X, y, build)
    if len(yo) == 0 or yo.sum() < 2:
        return None
    ic, sl = cal_intercept_slope(yo, po)
    base, lift_mean, lift_lo, lift_hi = paired_bootstrap_lift(yo, po)
    met = {"estimator": name, "validation": "expanding-window rolling-origin (quarterly) pooled OOF",
           "oof_n": int(len(yo)), "oof_pos": int(yo.sum()), "test_folds": len(per_fold),
           "pr_auc": round(float(average_precision_score(yo, po)), 4), "pr_auc_no_skill": base,
           "pr_auc_lift_vs_baseline_mean": lift_mean, "pr_auc_lift_ci95": [lift_lo, lift_hi],
           "roc_auc": round(float(roc_auc_score(yo, po)), 4),
           "brier": round(float(brier_score_loss(yo, po)), 4),
           "log_loss": round(float(log_loss(yo, po, labels=[0, 1])), 4),
           "calib_intercept": round(ic, 3), "calib_slope": round(sl, 3),
           "per_fold": per_fold}
    return met


def main():
    print("Loading + labeling (read-only)...")
    df = load_data()
    y = df["Label"].to_numpy().astype(int)
    n_pos_groups = df.loc[y == 1, "hMy"].nunique()
    print(f"  rows={len(df)} positives={int(y.sum())} converting_prospects={n_pos_groups} base={y.mean():.4%}")

    feats = select_features(df, y)
    print(f"  elastic-net selected {len(feats)} features: {feats}")
    X, med = matrix(df, feats)

    pen = evaluate("penalized-logistic (ridge)", df, X, y, _penalized)
    fir = evaluate("firth-flic", df, X, y, lambda: FirthLogisticRegression(flic=True))
    print("\n  PENALIZED:", json.dumps({k: v for k, v in (pen or {}).items() if k != "per_fold"}))
    print("  FIRTH   :", json.dumps({k: v for k, v in (fir or {}).items() if k != "per_fold"}))

    # promote by pooled OOF PR-AUC, tie-break by |calib_slope-1|
    cands = [m for m in (pen, fir) if m]
    if not cands:
        print("  no valid model (too few test positives). Abort."); return
    winner = max(cands, key=lambda m: (m["pr_auc"], -abs(m["calib_slope"] - 1)))
    print(f"\n  WINNER: {winner['estimator']} (pooled OOF PR-AUC {winner['pr_auc']} vs no-skill {winner['pr_auc_no_skill']})")

    # fit final winner on ALL data for serving
    if winner["estimator"].startswith("firth"):
        model = FirthLogisticRegression(flic=True).fit(X.values, y)
        calib = "firth-flic"
    else:
        try:
            model = CalibratedClassifierCV(_penalized(), method="sigmoid", cv=3).fit(X.values, y)
            calib = "platt"
        except Exception as e:
            print("  platt fallback:", str(e)[:100]); model = _penalized().fit(X.values, y); calib = "none"

    bundle = {
        "model": model, "feature_columns": feats, "model_input_features": feats,
        "feature_medians": med, "bundle_format": "corrected-logistic",
        "thresholds": {"risk_low_max": 0.33, "risk_medium_max": 0.66},
        "calibration": {"method": calib, "prob_floor": 0.002, "prob_ceil": 0.99},
        "metrics": {"winner": winner, "penalized": pen, "firth": fir,
                    "n_converting_prospects": int(n_pos_groups), "n_prospects": int(df["hMy"].nunique()),
                    "n_rows": int(len(df)), "n_pos_rows": int(y.sum()),
                    "base_rate": round(float(y.mean()), 4),
                    "selected_features": feats, "feature_pool": len(KEEP)},
        "version": VERSION, "trained_at": TS, "algorithm": winner["estimator"], "horizon_days": 180,
        "label": "corrected bridge: vw_Deal.tenantId->CommAmendmentSnapshot.hTenant, iType 0/1/2/15, "
                 "COALESCE(dtSign,dtStart) within 180d after snapshot",
        "source": "real-corrected-label",
    }
    manifest = {"contract": "signal-desk-brokerage", "version": VERSION, "feature_count": len(feats),
                "source": "real-corrected-label", "estimator": winner["estimator"],
                "trained_at": TS, "metrics": {k: winner[k] for k in
                ("pr_auc", "pr_auc_no_skill", "pr_auc_lift_ci95", "roc_auc", "brier", "calib_intercept",
                 "calib_slope", "oof_pos", "test_folds")},
                "note": "Small-n-defensible: elastic-net selection + (penalized vs Firth+FLIC) + "
                        "rolling-origin walk-forward + paired-bootstrap CI, no resampling. Replaces "
                        "v0.2 LightGBM. INTERIM/data-limited until historical signal backfill (Revelio/"
                        "Crunchbase/Form D/press) lifts the converting-prospect count."}
    d = BUNDLES / "signal_desk_brokerage" / VERSION
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, d / "bundle.joblib")
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    if os.environ.get("GBME_NO_PROMOTE"):
        print(f"  saved candidate {VERSION} (calibration={calib}); default.txt NOT changed (pipeline owns promotion)")
    else:
        (BUNDLES / "signal_desk_brokerage" / "default.txt").write_text(VERSION)
        print(f"  saved {VERSION} (calibration={calib}); default.txt -> {VERSION}")
    print("DONE")


if __name__ == "__main__":
    main()
