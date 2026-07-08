"""Brokerage deal-win model (Model A) — port + graduation test of the Signal Desk
shadow win-model, with the NEW point-in-time scoop features the 2022-2026 zi.Scoop
backfill unlocked.

Deal grain (rea.YardiDeal). Label: StageResolved='Executed' -> won; Cancelled / Dead Deal /
DeadDealReason -> lost; else in-progress (excluded). ~906 resolved, 24.7% win.
Features (parity w/ api/train-win-model.mjs): cohortRate (k=12 shrink, TRAIN-ONLY in
backtest), isNew/isRenew/isExpansion, sqft band (micro<2.5k<small<10k<mid<40k<large),
leadBroker. NEW: point-in-time scoop features (OriginalPublishedDate <= deal-open) via
ZiCompanyId.

Honest harness: corrected resolved-date split (ModifiedAt), cohort prior rebuilt train-only,
paired-bootstrap CI of AUC lift vs the cohort-rate baseline, ECE. Compares base vs
scoop-enriched. Reports the graduation verdict; saves a bundle + metrics (SHADOW until it
clears the gate). SELECT-only; writes only the local bundle.

Run: PYTHONPATH=. .venv/bin/python scripts/train_brokerage_dealwin.py
"""
from __future__ import annotations
import json, sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core import db                                          # noqa: E402

BUNDLES = ROOT / "bundles"
TS = datetime.now().strftime("%Y%m%d_%H%M%S")
VERSION = "v0.1-dealwin-shadow"
SPLITS = ["2025-06-01", "2025-09-01", "2025-11-01"]   # resolved-date splits, per the app doc
RNG = np.random.default_rng(12345)
BASE_FEATS = ["cohortRate", "isNew", "isRenew", "isExpansion",
              "bandMicro", "bandSmall", "bandMid", "bandLarge", "leadBroker"]
SCOOP_FEATS = ["scoop_cnt_180d", "scoop_cnt_365d", "days_since_scoop",
               "had_funding_365d", "had_hiring_365d", "had_layoff_180d", "has_zi"]


def band_of(area):
    if area is None or not np.isfinite(area) or area <= 0:
        return None
    return "micro" if area < 2500 else "small" if area < 10000 else "mid" if area < 40000 else "large"


def norm_type(dt):
    s = ("" if dt is None else str(dt)).strip().lower()
    return "renew" if s.startswith("renew") else "expansion" if s.startswith("expan") else "new" if s.startswith("new") else "other"


def load_deals() -> pd.DataFrame:
    d = db.query(
        "SELECT DealId, DealType, LeadSource, StageResolved, DeadDealReason, CreatedAt, ModifiedAt, "
        "COALESCE(NULLIF(EconArea,0), DesiredAreaMinSf) AS area, ZiCompanyId "
        "FROM [{}].rea.YardiDeal "
        "WHERE StageResolved='Executed' OR StageResolved IN ('Cancelled','Dead Deal') "
        "OR DeadDealReason IS NOT NULL".format(db.DB_MAIN))
    d["y"] = (d["StageResolved"].astype(str) == "Executed").astype(int)
    d["CreatedAt"] = pd.to_datetime(d["CreatedAt"])
    d["ModifiedAt"] = pd.to_datetime(d["ModifiedAt"])
    d["area"] = pd.to_numeric(d["area"], errors="coerce")
    d["band"] = d["area"].apply(band_of)
    d["dt"] = d["DealType"].apply(norm_type)
    d["leadBroker"] = d["LeadSource"].astype(str).str.lower().str.contains("broker").astype(int)
    d["zi"] = pd.to_numeric(d["ZiCompanyId"], errors="coerce")
    return d


def load_scoops(company_ids):
    ids = [int(c) for c in company_ids if pd.notna(c)]
    if not ids:
        return pd.DataFrame(columns=["CompanyID", "dt", "stype"])
    out = []
    for i in range(0, len(ids), 500):
        chunk = ",".join(str(x) for x in ids[i:i + 500])
        try:
            df = db.query(
                f"SELECT s.CompanyID, s.OriginalPublishedDate AS dt, st.Name AS stype "
                f"FROM [{db.DB_MAIN}].zi.Scoop s "
                f"LEFT JOIN [{db.DB_MAIN}].zi.ScoopTypeMap m ON m.ScoopID=s.ScoopID "
                f"LEFT JOIN [{db.DB_MAIN}].zi.ScoopType st ON st.TypeID=m.TypeID "
                f"WHERE s.CompanyID IN ({chunk}) AND s.OriginalPublishedDate IS NOT NULL")
        except Exception:
            df = db.query(
                f"SELECT s.CompanyID, s.OriginalPublishedDate AS dt, CAST(NULL AS varchar(50)) AS stype "
                f"FROM [{db.DB_MAIN}].zi.Scoop s "
                f"WHERE s.CompanyID IN ({chunk}) AND s.OriginalPublishedDate IS NOT NULL")
        out.append(df)
    s = pd.concat(out, ignore_index=True)
    s["dt"] = pd.to_datetime(s["dt"])
    s["CompanyID"] = pd.to_numeric(s["CompanyID"], errors="coerce")
    s["stype"] = s["stype"].astype(str).str.lower()
    return s


def scoop_features(deals, scoops):
    """Per-deal point-in-time scoop features (strictly OriginalPublishedDate <= deal-open)."""
    by_co = {cid: g.sort_values("dt") for cid, g in scoops.groupby("CompanyID")}
    feats = []
    for r in deals.itertuples(index=False):
        f = {k: 0.0 for k in SCOOP_FEATS}
        co = r.zi
        if pd.notna(co) and int(co) in by_co:
            f["has_zi"] = 1.0
            g = by_co[int(co)]
            past = g[g["dt"] <= r.CreatedAt]
            if len(past):
                d180 = r.CreatedAt - pd.Timedelta(days=180)
                d365 = r.CreatedAt - pd.Timedelta(days=365)
                f["scoop_cnt_180d"] = float((past["dt"] > d180).sum())
                f["scoop_cnt_365d"] = float((past["dt"] > d365).sum())
                f["days_since_scoop"] = float((r.CreatedAt - past["dt"].max()).days)
                typ = past["stype"]
                f["had_funding_365d"] = float(((past["dt"] > d365) & typ.str.contains("fund")).any())
                f["had_hiring_365d"] = float(((past["dt"] > d365) & typ.str.contains("hir")).any())
                f["had_layoff_180d"] = float(((past["dt"] > d180) & typ.str.contains("layoff|workforce|cut")).any())
            else:
                f["days_since_scoop"] = 9999.0
        else:
            f["days_since_scoop"] = 9999.0
        feats.append(f)
    return pd.DataFrame(feats, index=deals.index)


def shrink(won, lost, g, k=12):
    return (won + k * g) / (won + lost + k)


def cohort_fn(train):
    gw, gl = int(train["y"].sum()), int((1 - train["y"]).sum())
    g = gw / (gw + gl) if (gw + gl) else 0.25
    cc = {}
    for b, grp in train.groupby("band"):
        cc[b] = (int(grp["y"].sum()), int((1 - grp["y"]).sum()))
    return lambda band: shrink(*(cc.get(band, (gw, gl))), g=g), g


def base_matrix(deals, cr):
    X = pd.DataFrame(index=deals.index)
    X["cohortRate"] = [cr(b) for b in deals["band"]]
    X["isNew"] = (deals["dt"] == "new").astype(float)
    X["isRenew"] = (deals["dt"] == "renew").astype(float)
    X["isExpansion"] = (deals["dt"] == "expansion").astype(float)
    for b in ("micro", "small", "mid", "large"):
        X["band" + b.capitalize()] = (deals["band"] == b).astype(float)
    X["leadBroker"] = deals["leadBroker"].astype(float)
    return X


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.sum():
            e += abs(p[m].mean() - y[m].mean()) * m.sum() / len(y)
    return float(e)


def fit_lr(X, y):
    return Pipeline([("sc", StandardScaler()),
                     ("lr", LogisticRegression(C=1.0, max_iter=2000, random_state=7))]).fit(X.values, y)


def backtest(deals, scoopX, feats, split, use_scoops):
    tr, te = deals[deals["ModifiedAt"] < split], deals[deals["ModifiedAt"] >= split]
    if len(te) < 20 or te["y"].sum() < 5 or tr["y"].sum() < 10:
        return None
    cr, _ = cohort_fn(tr)
    def mat(dd):
        X = base_matrix(dd, cr)
        if use_scoops:
            X = pd.concat([X, scoopX.loc[dd.index]], axis=1)
        return X[feats]
    m = fit_lr(mat(tr), tr["y"].to_numpy())
    pte = m.predict_proba(mat(te).values)[:, 1]
    yte = te["y"].to_numpy()
    base = np.array([cr(b) for b in te["band"]])         # cohort-rate-alone baseline
    diffs = []
    for _ in range(2000):
        idx = RNG.integers(0, len(yte), len(yte))
        if yte[idx].sum() < 2 or yte[idx].sum() == len(idx):
            continue
        diffs.append(roc_auc_score(yte[idx], pte[idx]) - roc_auc_score(yte[idx], base[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"split": split, "test_n": int(len(te)), "test_win": round(float(yte.mean()), 3),
            "model_auc": round(float(roc_auc_score(yte, pte)), 3),
            "baseline_auc": round(float(roc_auc_score(yte, base)), 3),
            "auc_diff_mean": round(float(np.mean(diffs)), 3),
            "auc_diff_ci95": [round(float(lo), 3), round(float(hi), 3)],
            "model_brier": round(float(brier_score_loss(yte, pte)), 3),
            "model_ece": round(float(ece(yte, pte)), 3),
            "leadbroker_tr": round(float(tr["leadBroker"].mean()), 3),
            "leadbroker_te": round(float(te["leadBroker"].mean()), 3)}


def main():
    print("Loading resolved deals (read-only)...")
    deals = load_deals()
    print(f"  resolved={len(deals)} won={int(deals['y'].sum())} winrate={deals['y'].mean():.3f} "
          f"with_zi={int(deals['zi'].notna().sum())}")
    print("Loading scoops + building point-in-time features...")
    scoops = load_scoops(deals.loc[deals["zi"].notna(), "zi"].unique())
    scoopX = scoop_features(deals, scoops)
    print(f"  scoop rows={len(scoops)} | deals with >=1 prior scoop={int((scoopX['scoop_cnt_365d']>0).sum())}")

    allfeats = BASE_FEATS + SCOOP_FEATS
    res = {"base": [], "scoops": []}
    for sp in SPLITS:
        rb = backtest(deals, scoopX, BASE_FEATS, sp, False)
        rs = backtest(deals, scoopX, allfeats, sp, True)
        if rb: res["base"].append(rb)
        if rs: res["scoops"].append(rs)
        if rb and rs:
            print(f"\n  split {sp}: base AUC {rb['model_auc']} (vs cohort {rb['baseline_auc']}, "
                  f"diff {rb['auc_diff_mean']} CI{rb['auc_diff_ci95']}) | "
                  f"scoops AUC {rs['model_auc']} diff {rs['auc_diff_mean']} CI{rs['auc_diff_ci95']} | "
                  f"ECE base {rb['model_ece']}/scoops {rs['model_ece']}")

    def avg(rows, k): return round(float(np.mean([r[k] for r in rows])), 3) if rows else None
    summary = {
        "resolved_deals": int(len(deals)), "won": int(deals["y"].sum()),
        "win_rate": round(float(deals["y"].mean()), 3),
        "with_zicompany": int(deals["zi"].notna().sum()),
        "splits": SPLITS,
        "base_mean_auc": avg(res["base"], "model_auc"), "base_mean_diff": avg(res["base"], "auc_diff_mean"),
        "scoops_mean_auc": avg(res["scoops"], "model_auc"), "scoops_mean_diff": avg(res["scoops"], "auc_diff_mean"),
        "baseline_mean_auc": avg(res["base"], "baseline_auc"),
        "per_split_base": res["base"], "per_split_scoops": res["scoops"],
    }
    # graduation read: AUC-lift CI excludes 0 on the larger splits?
    grad = all(r["auc_diff_ci95"][0] > 0 for r in res["scoops"][:2]) if len(res["scoops"]) >= 2 else False
    summary["graduation_auc_gate_2of2_larger_splits"] = bool(grad)
    print("\n  SUMMARY:", json.dumps({k: v for k, v in summary.items() if not k.startswith("per_split")}))
    print("  scoops help ranking?",
          "yes" if (summary["scoops_mean_diff"] or 0) > (summary["base_mean_diff"] or 0) else "no/negligible")

    # final model trained on ALL resolved deals (full-data cohort prior) for the saved bundle
    cr_all, _ = cohort_fn(deals)
    Xall = pd.concat([base_matrix(deals, cr_all), scoopX], axis=1)[allfeats]
    final = fit_lr(Xall, deals["y"].to_numpy())
    bundle = {"model": final, "feature_columns": allfeats, "bundle_format": "dealwin-logistic",
              "calibration": {"method": "none (shadow)", "prob_floor": 0.01, "prob_ceil": 0.95},
              "metrics": summary, "version": VERSION, "trained_at": TS, "algorithm": "l2-logistic",
              "grain": "deal", "label": "StageResolved=Executed (won) vs Cancelled/Dead Deal (lost)",
              "source": "rea.YardiDeal resolved deals", "status": "shadow"}
    d = BUNDLES / "signal_desk_dealwin" / VERSION
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, d / "bundle.joblib")
    (d / "manifest.json").write_text(json.dumps({"contract": "signal-desk-dealwin", "version": VERSION,
        "grain": "deal", "resolved_deals": summary["resolved_deals"], "win_rate": summary["win_rate"],
        "metrics": summary, "status": "shadow",
        "note": "Deal re-ranker ported from the Signal Desk shadow win-model + point-in-time scoop "
                "features (2022-2026 backfill). SHADOW until it clears the AUC-lift graduation gate vs "
                "live P(win) with leadBroker ablation + temporal-split integrity."}, indent=2, default=str))
    print(f"  saved {VERSION} bundle (NOT promoted; shadow). DONE")


if __name__ == "__main__":
    main()
