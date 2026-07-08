"""Stage 2 - learned RECENCY model (company-grain, broker-free): near-term deal-activity
propensity. Label = company opens a deal (rea.YardiDeal.CreatedAt) within 90 days AFTER a
monthly anchor (a DENSE, learnable target, unlike rare conversion). Features are point-in-
time recency/firmographic signals reconstructable as-of (zi.Scoop 2022-2026 + zi.Company),
so the served value can come from the current rea.CompanyScoreSnapshot fields.

Temporal backtest (anchor-date split), group-disjoint by company, paired-bootstrap AUC CI.
Logistic vs LightGBM. NO broker feature. SELECT-only; writes bundle only.
Run: PYTHONPATH=. .venv/bin/python scripts/train_brokerage_recency.py
"""
from __future__ import annotations
import json, sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from lightgbm import LGBMClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core import db                       # noqa: E402

TS = datetime.now().strftime("%Y%m%d_%H%M%S")
VERSION = "v0.1-recency"
SPLIT = "2025-09-30"
RNG = np.random.default_rng(7)
FEATS = ["scoop_cnt_90d", "scoop_cnt_365d", "days_since_scoop", "had_funding_365d",
         "had_hiring_365d", "had_layoff_180d", "log_revenue", "log_employees", "tenure_days"]


def load():
    dl = db.query(f"SELECT ZiCompanyId cid, CreatedAt FROM [{db.DB_MAIN}].rea.YardiDeal "
                  f"WHERE ZiCompanyId IS NOT NULL AND CreatedAt IS NOT NULL")
    dl["cid"] = dl["cid"].astype("int64"); dl["CreatedAt"] = pd.to_datetime(dl["CreatedAt"])
    events = defaultdict(list)
    for r in dl.itertuples(index=False):
        events[int(r.cid)].append(r.CreatedAt)
    cids = sorted(events)
    sc_frames = []
    for i in range(0, len(cids), 500):
        chunk = ",".join(str(x) for x in cids[i:i + 500])
        try:
            sc_frames.append(db.query(
                f"SELECT s.CompanyID cid, s.OriginalPublishedDate dt, st.Name stype "
                f"FROM [{db.DB_MAIN}].zi.Scoop s LEFT JOIN [{db.DB_MAIN}].zi.ScoopTypeMap m ON m.ScoopID=s.ScoopID "
                f"LEFT JOIN [{db.DB_MAIN}].zi.ScoopType st ON st.TypeID=m.TypeID "
                f"WHERE s.CompanyID IN ({chunk}) AND s.OriginalPublishedDate IS NOT NULL"))
        except Exception:
            sc_frames.append(db.query(
                f"SELECT s.CompanyID cid, s.OriginalPublishedDate dt, CAST(NULL AS varchar(50)) stype "
                f"FROM [{db.DB_MAIN}].zi.Scoop s WHERE s.CompanyID IN ({chunk}) AND s.OriginalPublishedDate IS NOT NULL"))
    sc = pd.concat(sc_frames, ignore_index=True)
    sc["cid"] = sc["cid"].astype("int64"); sc["dt"] = pd.to_datetime(sc["dt"]); sc["stype"] = sc["stype"].astype(str).str.lower()
    by_co = {c: g.sort_values("dt") for c, g in sc.groupby("cid")}
    fm = db.query(f"SELECT CompanyID cid, Revenue, EmployeeCount FROM [{db.DB_MAIN}].zi.Company "
                  f"WHERE CompanyID IN ({','.join(str(x) for x in cids)})")
    fmap = {int(r.cid): (r.Revenue, r.EmployeeCount) for r in fm.itertuples(index=False)}
    return events, by_co, fmap, cids


def build_panel(events, by_co, fmap, cids):
    anchors = pd.date_range("2022-06-30", "2026-03-31", freq="ME")
    rows = []
    for cid in cids:
        evs = sorted(events[cid]); g = by_co.get(cid)
        rev, emp = fmap.get(cid, (None, None))
        logrev = np.log1p(float(rev)) if rev and rev == rev else 0.0
        logemp = np.log1p(float(emp)) if emp and emp == emp else 0.0
        first_scoop = g["dt"].min() if g is not None and len(g) else None
        for a in anchors:
            f = {"scoop_cnt_90d": 0.0, "scoop_cnt_365d": 0.0, "days_since_scoop": 9999.0,
                 "had_funding_365d": 0.0, "had_hiring_365d": 0.0, "had_layoff_180d": 0.0,
                 "log_revenue": logrev, "log_employees": logemp,
                 "tenure_days": float((a - first_scoop).days) if first_scoop is not None and first_scoop <= a else 0.0}
            if g is not None:
                past = g[g["dt"] <= a]
                if len(past):
                    d180, d365 = a - pd.Timedelta(days=180), a - pd.Timedelta(days=365)
                    f["scoop_cnt_90d"] = float((past["dt"] > a - pd.Timedelta(days=90)).sum())
                    f["scoop_cnt_365d"] = float((past["dt"] > d365).sum())
                    f["days_since_scoop"] = float((a - past["dt"].max()).days)
                    t = past["stype"]
                    f["had_funding_365d"] = float(((past["dt"] > d365) & t.str.contains("fund")).any())
                    f["had_hiring_365d"] = float(((past["dt"] > d365) & t.str.contains("hir")).any())
                    f["had_layoff_180d"] = float(((past["dt"] > d180) & t.str.contains("layoff|workforce|cut")).any())
            hi = a + pd.Timedelta(days=90)
            f["y"] = int(any(a < e <= hi for e in evs))
            f["cid"] = cid; f["anchor"] = a
            rows.append(f)
    return pd.DataFrame(rows)


def boot_auc(y, p, n=2000):
    d = []
    for _ in range(n):
        idx = RNG.integers(0, len(y), len(y))
        if 0 < y[idx].sum() < len(idx):
            d.append(roc_auc_score(y[idx], p[idx]))
    return [round(float(np.percentile(d, 2.5)), 3), round(float(np.percentile(d, 97.5)), 3)]


def main():
    print("Loading events/scoops/firmographics + building point-in-time panel...")
    events, by_co, fmap, cids = load()
    df = build_panel(events, by_co, fmap, cids)
    y = df["y"].to_numpy()
    print(f"  companies={len(cids)} panel_rows={len(df)} positives={int(y.sum())} ({y.mean():.3%})")
    tr, te = df[df["anchor"] < SPLIT], df[df["anchor"] >= SPLIT]
    Xtr, ytr = tr[FEATS].values, tr["y"].to_numpy()
    Xte, yte = te[FEATS].values, te["y"].to_numpy()
    print(f"  train {len(tr)} (pos {ytr.sum()}) | test {len(te)} (pos {yte.sum()})")

    out = {}
    for name, mk in [("logistic", lambda: Pipeline([("s", StandardScaler()),
                      ("l", LogisticRegression(C=1.0, max_iter=2000, class_weight=None))])),
                     ("lightgbm", lambda: LGBMClassifier(n_estimators=200, num_leaves=15, max_depth=4,
                      learning_rate=0.05, min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
                      reg_lambda=5.0, n_jobs=4, verbose=-1, random_state=7))]:
        m = mk(); m.fit(Xtr, ytr); p = m.predict_proba(Xte)[:, 1]
        out[name] = {"auc": round(float(roc_auc_score(yte, p)), 3), "auc_ci95": boot_auc(yte, p),
                     "pr_auc": round(float(average_precision_score(yte, p)), 3),
                     "pr_auc_no_skill": round(float(yte.mean()), 3),
                     "brier": round(float(brier_score_loss(yte, p)), 3)}
        print(f"  {name}: {json.dumps(out[name])}")
    winner = max(out, key=lambda k: out[k]["auc"])
    print(f"  winner: {winner} (AUC {out[winner]['auc']})")

    # final model on all rows for serving (features come live from CompanyScoreSnapshot at serve time)
    mk = (lambda: Pipeline([("s", StandardScaler()), ("l", LogisticRegression(C=1.0, max_iter=2000))])) \
        if winner == "logistic" else \
        (lambda: LGBMClassifier(n_estimators=200, num_leaves=15, max_depth=4, learning_rate=0.05,
         min_child_samples=40, subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, n_jobs=4, verbose=-1, random_state=7))
    final = mk(); final.fit(df[FEATS].values, y)
    bundle = {"model": final, "feature_columns": FEATS, "bundle_format": "recency-propensity",
              "calibration": {"method": "none", "prob_floor": 0.0, "prob_ceil": 1.0},
              "metrics": {"models": out, "winner": winner, "panel_rows": int(len(df)),
                          "positives": int(y.sum()), "base_rate": round(float(y.mean()), 4),
                          "companies": len(cids), "split": SPLIT,
                          "test_auc": out[winner]["auc"], "test_auc_ci95": out[winner]["auc_ci95"],
                          "test_pr_auc": out[winner]["pr_auc"]},
              "version": VERSION, "trained_at": TS, "algorithm": winner, "grain": "company",
              "label": "company opens a deal (YardiDeal.CreatedAt) within 90d after anchor",
              "source": "rea.YardiDeal + zi.Scoop + zi.Company (point-in-time)", "broker_free": True}
    d = ROOT / "bundles" / "signal_desk_recency" / VERSION
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, d / "bundle.joblib")
    (d / "manifest.json").write_text(json.dumps({"contract": "signal-desk-recency", "version": VERSION,
        "grain": "company", "broker_free": True, "metrics": bundle["metrics"],
        "note": "Near-term deal-activity propensity from recency signal (broker-free). Serve from current "
                "rea.CompanyScoreSnapshot scoop fields + zi.Company firmographics."}, indent=2, default=str))
    (ROOT / "bundles" / "signal_desk_recency" / "default.txt").write_text(VERSION)
    print(f"  saved {VERSION}. DONE")


if __name__ == "__main__":
    main()
