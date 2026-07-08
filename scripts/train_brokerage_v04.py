"""Brokerage cold-prospect v0.4 — v0.3 corrected-bridge label + scoop features RECOMPUTED
from raw zi.Scoop (2022-2026 backfill) via rea.ProspectCompanyLink (the snapshot scoop
columns are stale). Tests whether the now-available scoop history helps the cold-prospect
model. Same small-n stack as v0.3 (elastic-net selection + penalized vs Firth+FLIC,
rolling-origin + paired-bootstrap CI). Promote only if it beats v0.3 (PR-AUC ~0.108).

Run: PYTHONPATH=. .venv/bin/python scripts/train_brokerage_v04.py    (SELECT-only; writes bundle only)
"""
from __future__ import annotations
import json, sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from core import db                                   # noqa: E402
import train_brokerage_v03 as v3                      # noqa: E402
from firth_logistic import FirthLogisticRegression    # noqa: E402

TS = datetime.now().strftime("%Y%m%d_%H%M%S")
VERSION = "v0.4-corrected-scoops"
SCOOP_FEATS = ["scoop_cnt_180d", "scoop_cnt_365d", "days_since_scoop",
               "had_funding_365d", "had_hiring_365d", "had_layoff_180d", "has_zi"]


def scoop_matrix(df):
    link = db.query(f"SELECT hMy, CompanyID FROM [{db.DB_MAIN}].rea.ProspectCompanyLink WHERE CompanyID IS NOT NULL")
    h2c = {int(r.hMy): int(r.CompanyID) for r in link.itertuples(index=False)}
    cids = sorted(set(h2c.values()))
    frames = []
    for i in range(0, len(cids), 500):
        chunk = ",".join(str(x) for x in cids[i:i + 500])
        try:
            frames.append(db.query(
                f"SELECT s.CompanyID, s.OriginalPublishedDate dt, st.Name stype "
                f"FROM [{db.DB_MAIN}].zi.Scoop s LEFT JOIN [{db.DB_MAIN}].zi.ScoopTypeMap m ON m.ScoopID=s.ScoopID "
                f"LEFT JOIN [{db.DB_MAIN}].zi.ScoopType st ON st.TypeID=m.TypeID "
                f"WHERE s.CompanyID IN ({chunk}) AND s.OriginalPublishedDate IS NOT NULL"))
        except Exception:
            frames.append(db.query(
                f"SELECT s.CompanyID, s.OriginalPublishedDate dt, CAST(NULL AS varchar(50)) stype "
                f"FROM [{db.DB_MAIN}].zi.Scoop s WHERE s.CompanyID IN ({chunk}) AND s.OriginalPublishedDate IS NOT NULL"))
    sc = pd.concat(frames, ignore_index=True)
    sc["dt"] = pd.to_datetime(sc["dt"]); sc["CompanyID"] = sc["CompanyID"].astype("int64")
    sc["stype"] = sc["stype"].astype(str).str.lower()
    by_co = {cid: g.sort_values("dt") for cid, g in sc.groupby("CompanyID")}
    rows = []
    for r in df.itertuples(index=False):
        f = {k: 0.0 for k in SCOOP_FEATS}; f["days_since_scoop"] = 9999.0
        cid = h2c.get(int(r.hMy))
        if cid is not None and cid in by_co:
            f["has_zi"] = 1.0
            g = by_co[cid]; past = g[g["dt"] <= r.SnapshotDate]
            if len(past):
                d180, d365 = r.SnapshotDate - pd.Timedelta(days=180), r.SnapshotDate - pd.Timedelta(days=365)
                f["scoop_cnt_180d"] = float((past["dt"] > d180).sum())
                f["scoop_cnt_365d"] = float((past["dt"] > d365).sum())
                f["days_since_scoop"] = float((r.SnapshotDate - past["dt"].max()).days)
                t = past["stype"]
                f["had_funding_365d"] = float(((past["dt"] > d365) & t.str.contains("fund")).any())
                f["had_hiring_365d"] = float(((past["dt"] > d365) & t.str.contains("hir")).any())
                f["had_layoff_180d"] = float(((past["dt"] > d180) & t.str.contains("layoff|workforce|cut")).any())
        rows.append(f)
    return pd.DataFrame(rows, index=df.index), len(sc), len(h2c)


def main():
    print("Loading v0.3 base (label + 29 feats) + recomputing scoop features from raw...")
    df = v3.load_data()
    y = df["Label"].to_numpy().astype(int)
    X29, med = v3.matrix(df, v3.KEEP)
    scoopX, n_scoops, n_linked = scoop_matrix(df)
    print(f"  rows={len(df)} converting_prospects={df.loc[y==1,'hMy'].nunique()} | "
          f"linked={n_linked} scoop_rows={n_scoops} | snap w/ prior scoop={int((scoopX['scoop_cnt_365d']>0).sum())}")
    Xaug = pd.concat([X29, scoopX], axis=1)
    pool = v3.KEEP + SCOOP_FEATS

    # elastic-net selection over the AUGMENTED pool (allow up to 10 given the added family)
    Z = StandardScaler().fit_transform(Xaug.values)
    enet = LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, C=0.2, max_iter=5000, random_state=7)
    enet.fit(Z, y); coef = np.abs(enet.coef_[0])
    sel = [pool[i] for i in np.argsort(coef)[::-1] if coef[i] > 1e-8][:10]
    n_scoop_sel = sum(1 for f in sel if f in SCOOP_FEATS)
    print(f"  selected {len(sel)} ({n_scoop_sel} scoop) -> {sel}")

    pen = v3.evaluate("penalized", df, Xaug[sel], y, v3._penalized)
    fir = v3.evaluate("firth", df, Xaug[sel], y, lambda: FirthLogisticRegression(flic=True))
    print("  PENALIZED:", json.dumps({k: pen[k] for k in ("pr_auc", "pr_auc_no_skill", "pr_auc_lift_ci95", "roc_auc")}))
    print("  FIRTH    :", json.dumps({k: fir[k] for k in ("pr_auc", "pr_auc_no_skill", "pr_auc_lift_ci95", "roc_auc")}))
    print(f"  v0.3 reference PR-AUC = 0.108 (scoop-free).  v0.4 best = {max(pen['pr_auc'], fir['pr_auc'])}")
    print("  scoops help cold-prospect?",
          "YES" if max(pen["pr_auc"], fir["pr_auc"]) > 0.115 else "no/negligible (keep v0.3)")

    # save v0.4 bundle regardless (for the record); promotion decision is reported, not auto-applied
    w = max((pen, fir), key=lambda m: m["pr_auc"])
    if w["estimator"].startswith("firth"):
        model = FirthLogisticRegression(flic=True).fit(Xaug[sel].values, y); calib = "firth-flic"
    else:
        model = CalibratedClassifierCV(v3._penalized(), method="sigmoid", cv=3).fit(Xaug[sel].values, y); calib = "platt"
    bundle = {"model": model, "feature_columns": sel, "feature_medians": med,
              "scoop_features": [f for f in sel if f in SCOOP_FEATS],
              "bundle_format": "corrected-scoops", "calibration": {"method": calib, "prob_floor": 0.002, "prob_ceil": 0.99},
              "metrics": {"winner": w, "penalized": pen, "firth": fir, "selected_features": sel,
                          "n_scoop_selected": n_scoop_sel, "n_converting_prospects": int(df.loc[y==1,'hMy'].nunique()),
                          "n_prospects": int(df["hMy"].nunique()), "n_rows": int(len(df)),
                          "n_pos_rows": int(y.sum()), "base_rate": round(float(y.mean()), 4),
                          "v03_reference_pr_auc": 0.108, "feature_pool": len(pool)},
              "version": VERSION, "trained_at": TS, "algorithm": w["estimator"], "horizon_days": 180,
              "label": "corrected bridge (same as v0.3) + raw point-in-time scoops via ProspectCompanyLink",
              "source": "real-corrected-label", "requires_scoop_recompute": True}
    d = ROOT / "bundles" / "signal_desk_brokerage" / VERSION
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, d / "bundle.joblib")
    (d / "manifest.json").write_text(json.dumps({"contract": "signal-desk-brokerage", "version": VERSION,
        "feature_count": len(sel), "n_scoop_selected": n_scoop_sel, "metrics": w,
        "note": "v0.3 + raw point-in-time scoop features (zi.Scoop 2022-2026 via ProspectCompanyLink). "
                "Promote ONLY if it beats v0.3 AND a serving-time scoop recompute is wired (snapshot scoop "
                "columns are stale)."}, indent=2, default=str))
    print(f"  saved {VERSION} (NOT auto-promoted). DONE")


if __name__ == "__main__":
    main()
