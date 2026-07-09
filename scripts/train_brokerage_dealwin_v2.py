"""Brokerage deal-win model v0.2 — LEAKAGE-FREE rebuild of the shadow re-ranker.

Fixes vs v0.1 (train_brokerage_dealwin.py):
  * DROP leadBroker / LeadSource       (era/outcome proxy: Existing-Customer 97% win vs NULL 2%)
  * DROP EconArea / EconStartRent      (present on 99% of won vs 48% of lost -> presence leaks)
  * Restrict to CreatedAt >= 2024      (pre-2024 is ~all losses = capture/censoring artifact)
  * GROUP-DISJOINT by TenantId         (CommProspectId is null on ~86% of won deals; TenantId
                                        is populated -> ~214 distinct winning companies)
  * LightGBM w/ native missing handling (safe features are sparse: DealType 22%, area 13%,
                                        ZI-firmographics 38% but full when present)
  * Point-in-time features only (scoops OriginalPublishedDate <= deal-open CreatedAt)
  * Yardi-only vs Yardi+ZI ABLATION    (answers "does ZoomInfo earn its place at this grain?")

Honest harness: group-disjoint expanding-window rolling-origin walk-forward (split on ModifiedAt =
resolution date; a tenant never spans train/test), calibrated (Platt) pooled OOF, paired-bootstrap
AUC-lift CI vs the TRAIN-ONLY cohort-rate baseline, PR-AUC / Brier / ECE. SHADOW until it clears the
gate (lift CI excludes 0 on the larger folds AND beats the cohort baseline). SELECT-only; writes only
the local bundle.

Run: PYTHONPATH=. .venv/bin/python scripts/train_brokerage_dealwin_v2.py
"""
from __future__ import annotations
import json, sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core import db                                          # noqa: E402

BUNDLES = ROOT / "bundles"
TS = datetime.now().strftime("%Y%m%d_%H%M%S")
VERSION = "v0.2-dealwin-leakfree"
RNG = np.random.default_rng(12345)
# resolution-date (ModifiedAt) splits for expanding-window walk-forward
SPLITS = ["2025-04-01", "2025-08-01", "2025-12-01"]

YARDI_BASE = ["is_renew", "is_new", "is_expansion", "area_minsf_log"]   # columns on `deals`
YARDI_FEATS = ["cohort_rate"] + YARDI_BASE                              # cohort_rate added per-fold
ZI_FEATS = ["emp_log", "rev_log", "company_age", "has_zi",
            "scoop_cnt_365d", "days_since_scoop", "had_reloc_expansion_365d",
            "had_funding_365d", "had_hiring_365d", "has_scoop"]


def assemble(sub, feat_df, cr):
    """Full point-in-time feature matrix for a deal subset: per-fold train-only cohort_rate
    + the Yardi base cols (on `deals`) + the ZI cols (on `feat_df`)."""
    m = pd.DataFrame(index=sub.index)
    m["cohort_rate"] = [cr(b) for b in sub["band"]]
    for c in YARDI_BASE:
        m[c] = sub[c].to_numpy()
    for c in ZI_FEATS:
        m[c] = feat_df.loc[sub.index, c].to_numpy()
    return m

RESOLVED = ("(StageResolved='Executed' OR StageResolved IN ('Cancelled','Dead Deal') "
            "OR DeadDealReason IS NOT NULL)")


def load_deals() -> pd.DataFrame:
    d = db.query(
        f"SELECT DealId, TenantId, CommProspectId, DealType, StageResolved, DeadDealReason, "
        f"CreatedAt, ModifiedAt, DesiredAreaMinSf, ZiCompanyId "
        f"FROM [{db.DB_MAIN}].rea.YardiDeal "
        f"WHERE {RESOLVED} AND CreatedAt >= '2024-01-01'")
    d["y"] = (d["StageResolved"].astype(str) == "Executed").astype(int)
    d["CreatedAt"] = pd.to_datetime(d["CreatedAt"])
    d["ModifiedAt"] = pd.to_datetime(d["ModifiedAt"])
    d = d[d["TenantId"].notna()].copy()                 # group key must exist (drops 2 wins)
    d["group"] = d["TenantId"].astype("int64")
    d["minsf"] = pd.to_numeric(d["DesiredAreaMinSf"], errors="coerce")
    d.loc[d["minsf"] <= 0, "minsf"] = np.nan
    d["area_minsf_log"] = np.log1p(d["minsf"])
    dt = d["DealType"].astype(str).str.lower()
    d["is_renew"] = dt.str.startswith("renew").astype(float)
    d["is_new"] = dt.str.startswith("new").astype(float)
    d["is_expansion"] = dt.str.startswith("expan").astype(float)
    d["band"] = d["minsf"].apply(_band)
    d["zi"] = pd.to_numeric(d["ZiCompanyId"], errors="coerce")
    return d.reset_index(drop=True)


def _band(a):
    if a is None or not np.isfinite(a) or a <= 0:
        return "unknown"
    return "micro" if a < 2500 else "small" if a < 10000 else "mid" if a < 40000 else "large"


def load_firmographics(zi_ids) -> dict:
    ids = sorted({int(c) for c in zi_ids if pd.notna(c)})
    if not ids:
        return {}
    out = {}
    for i in range(0, len(ids), 800):
        chunk = ",".join(str(x) for x in ids[i:i + 800])
        df = db.query(
            f"SELECT CompanyID, EmployeeCount, Revenue, FoundedYear "
            f"FROM [{db.DB_MAIN}].zi.Company WHERE CompanyID IN ({chunk})")
        for r in df.to_dict("records"):
            out[int(r["CompanyID"])] = r
    return out


def load_scoops(zi_ids) -> dict:
    ids = sorted({int(c) for c in zi_ids if pd.notna(c)})
    if not ids:
        return {}
    parts = []
    for i in range(0, len(ids), 500):
        chunk = ",".join(str(x) for x in ids[i:i + 500])
        df = db.query(
            f"SELECT s.CompanyID, s.OriginalPublishedDate AS dt, m.TypeID "
            f"FROM [{db.DB_MAIN}].zi.Scoop s "
            f"LEFT JOIN [{db.DB_MAIN}].zi.ScoopTypeMap m ON m.ScoopID = s.ScoopID "
            f"WHERE s.CompanyID IN ({chunk}) AND s.OriginalPublishedDate IS NOT NULL")
        parts.append(df)
    s = pd.concat(parts, ignore_index=True)
    s["dt"] = pd.to_datetime(s["dt"])
    s["CompanyID"] = pd.to_numeric(s["CompanyID"], errors="coerce")
    s["TypeID"] = pd.to_numeric(s["TypeID"], errors="coerce")
    return {int(cid): g.sort_values("dt") for cid, g in s.groupby("CompanyID")}


def build_features(deals, firmo, scoops_by_co) -> pd.DataFrame:
    """All features are point-in-time as of the deal's CreatedAt (deal-open)."""
    rows = []
    for r in deals.itertuples(index=False):
        f = {}
        # --- ZI firmographics (slow-moving; treat as static, flag missingness) ---
        co = int(r.zi) if pd.notna(r.zi) else None
        fm = firmo.get(co) if co is not None else None
        f["has_zi"] = 1.0 if fm else 0.0
        emp = fm.get("EmployeeCount") if fm else None
        rev = fm.get("Revenue") if fm else None
        fy = fm.get("FoundedYear") if fm else None
        f["emp_log"] = float(np.log1p(emp)) if (emp and emp > 0) else np.nan
        f["rev_log"] = float(np.log1p(rev)) if (rev and rev > 0) else np.nan
        f["company_age"] = float(r.CreatedAt.year - fy) if (fy and fy > 1800) else np.nan
        # --- ZI scoops, strictly OriginalPublishedDate <= deal-open ---
        for k in ("scoop_cnt_365d", "had_reloc_expansion_365d", "had_funding_365d", "had_hiring_365d"):
            f[k] = 0.0
        f["has_scoop"] = 0.0
        f["days_since_scoop"] = np.nan
        g = scoops_by_co.get(co) if co is not None else None
        if g is not None:
            past = g[g["dt"] <= r.CreatedAt]
            if len(past):
                f["has_scoop"] = 1.0
                d365 = r.CreatedAt - pd.Timedelta(days=365)
                recent = past[past["dt"] > d365]
                f["scoop_cnt_365d"] = float(len(recent))
                f["days_since_scoop"] = float((r.CreatedAt - past["dt"].max()).days)
                t = recent["TypeID"]
                f["had_reloc_expansion_365d"] = float((t == 8).any())    # Facilities Relocation/Expansion
                f["had_funding_365d"] = float((t == 2).any())            # Funding
                f["had_hiring_365d"] = float((t == 11).any())            # Hiring Plans
        rows.append(f)
    return pd.DataFrame(rows, index=deals.index)


def shrink(won, lost, g, k=10):
    return (won + k * g) / (won + lost + k)


def cohort_map(train):
    g = float(train["y"].mean()) if len(train) else 0.34
    cc = {}
    for b, grp in train.groupby("band"):
        cc[b] = (int(grp["y"].sum()), int((1 - grp["y"]).sum()))
    return lambda band: shrink(*cc.get(band, (int(train["y"].sum()), int((1 - train["y"]).sum()))), g=g)


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.sum():
            e += abs(p[m].mean() - y[m].mean()) * m.sum() / len(y)
    return float(e)


def fit_lgbm(X, y):
    return lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.03, num_leaves=8, max_depth=3,
        min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=1.0, reg_lambda=2.0, random_state=7, n_jobs=2, verbose=-1).fit(X, y)


def walk_forward(deals, feat_df, feats, use_label="model"):
    """Group-disjoint (TenantId) expanding-window rolling-origin. Returns per-fold rows +
    pooled OOF (score, cohort-baseline, y) for calibration/PR-AUC."""
    folds, oof_p, oof_base, oof_y = [], [], [], []
    for sp in SPLITS:
        split = pd.Timestamp(sp)
        tr = deals[deals["ModifiedAt"] < split]
        te = deals[deals["ModifiedAt"] >= split]
        if len(tr) < 40 or tr["y"].sum() < 10:
            continue
        tr_groups = set(tr["group"])
        te = te[~te["group"].isin(tr_groups)]              # group-disjoint
        if len(te) < 20 or te["y"].sum() < 5:
            continue
        cr = cohort_map(tr)
        Xtr, Xte = assemble(tr, feat_df, cr)[feats], assemble(te, feat_df, cr)[feats]
        m = fit_lgbm(Xtr, tr["y"].to_numpy())
        p = m.predict_proba(Xte)[:, 1]
        y = te["y"].to_numpy()
        base = np.array([cr(b) for b in te["band"]])
        diffs = []
        for _ in range(2000):
            idx = RNG.integers(0, len(y), len(y))
            s = y[idx].sum()
            if s < 2 or s == len(idx):
                continue
            diffs.append(roc_auc_score(y[idx], p[idx]) - roc_auc_score(y[idx], base[idx]))
        lo, hi = (np.percentile(diffs, [2.5, 97.5]) if diffs else (np.nan, np.nan))
        folds.append({"split": sp, "test_n": int(len(te)), "test_wins": int(y.sum()),
                      "test_winrate": round(float(y.mean()), 3),
                      "model_auc": round(float(roc_auc_score(y, p)), 3),
                      "cohort_auc": round(float(roc_auc_score(y, base)), 3),
                      "auc_lift_mean": round(float(np.mean(diffs)), 3) if diffs else None,
                      "auc_lift_ci95": [round(float(lo), 3), round(float(hi), 3)],
                      "model_pr_auc": round(float(average_precision_score(y, p)), 3),
                      "cohort_pr_auc": round(float(average_precision_score(y, base)), 3)})
        oof_p += list(p); oof_base += list(base); oof_y += list(y)
    return folds, np.array(oof_p), np.array(oof_base), np.array(oof_y)


def summarize(tag, folds, p, base, y):
    if not folds:
        return {"tag": tag, "note": "no valid folds"}
    # Platt-calibrate pooled OOF
    cal = LogisticRegression(C=1e6, max_iter=1000).fit(p.reshape(-1, 1), y)
    pc = cal.predict_proba(p.reshape(-1, 1))[:, 1]
    return {"tag": tag, "folds": folds,
            "mean_model_auc": round(float(np.mean([f["model_auc"] for f in folds])), 3),
            "mean_cohort_auc": round(float(np.mean([f["cohort_auc"] for f in folds])), 3),
            "mean_auc_lift": round(float(np.mean([f["auc_lift_mean"] for f in folds if f["auc_lift_mean"] is not None])), 3),
            "pooled_pr_auc": round(float(average_precision_score(y, p)), 3),
            "pooled_base_rate": round(float(y.mean()), 3),
            "brier_raw": round(float(brier_score_loss(y, p)), 4),
            "brier_cal": round(float(brier_score_loss(y, pc)), 4),
            "ece_raw": round(ece(y, p), 3), "ece_cal": round(ece(y, pc), 3),
            "lift_ci_excludes_0_larger_folds": bool(
                len(folds) >= 2 and all(f["auc_lift_ci95"][0] > 0 for f in folds[:2]))}


def main():
    print("Loading resolved 2024+ deals (read-only)...")
    deals = load_deals()
    print(f"  resolved={len(deals)} won={int(deals['y'].sum())} winrate={deals['y'].mean():.3f} "
          f"| distinct tenants={deals['group'].nunique()} | with_zi={int(deals['zi'].notna().sum())}")
    print("Loading ZI firmographics + point-in-time scoops...")
    firmo = load_firmographics(deals["zi"])
    scoops = load_scoops(deals["zi"])
    feat_df = build_features(deals, firmo, scoops)
    print(f"  firmo companies={len(firmo)} | scoop companies={len(scoops)} | "
          f"deals w/ prior scoop={int((feat_df['has_scoop']>0).sum())}")

    results = {}
    for tag, feats in [("yardi_only", YARDI_FEATS), ("yardi_plus_zi", YARDI_FEATS + ZI_FEATS)]:
        folds, p, base, y = walk_forward(deals, feat_df, feats)
        results[tag] = summarize(tag, folds, p, base, y)
        s = results[tag]
        print(f"\n[{tag}] folds={len(s.get('folds',[]))} mean_model_auc={s.get('mean_model_auc')} "
              f"vs cohort {s.get('mean_cohort_auc')} | mean_auc_lift={s.get('mean_auc_lift')} | "
              f"pooled PR-AUC={s.get('pooled_pr_auc')} (base {s.get('pooled_base_rate')}) | "
              f"ECE cal={s.get('ece_cal')} | lift_CI>0 on larger folds={s.get('lift_ci_excludes_0_larger_folds')}")
        for f in s.get("folds", []):
            print(f"    {f['split']}: n={f['test_n']} wins={f['test_wins']} "
                  f"model_auc={f['model_auc']} cohort_auc={f['cohort_auc']} "
                  f"lift={f['auc_lift_mean']} CI{f['auc_lift_ci95']}")

    zi_helps = (results.get("yardi_plus_zi", {}).get("mean_auc_lift") or -9) > \
               (results.get("yardi_only", {}).get("mean_auc_lift") or -9)
    graduates = bool(results.get("yardi_plus_zi", {}).get("lift_ci_excludes_0_larger_folds") or
                     results.get("yardi_only", {}).get("lift_ci_excludes_0_larger_folds"))
    print(f"\nABLATION: ZoomInfo adds lift over Yardi-only? {'YES' if zi_helps else 'NO / negligible'}")
    print(f"GRADUATION (beats cohort baseline, lift CI>0 on larger folds)? {'YES' if graduates else 'NO -> SHADOW'}")

    # Final model on ALL resolved deals (full-data cohort prior); saved SHADOW.
    best = "yardi_plus_zi" if zi_helps else "yardi_only"
    feats = (YARDI_FEATS + ZI_FEATS) if best == "yardi_plus_zi" else YARDI_FEATS
    cr_all = cohort_map(deals)
    Xall = assemble(deals, feat_df, cr_all)[feats]
    model = fit_lgbm(Xall, deals["y"].to_numpy())
    cal = LogisticRegression(C=1e6, max_iter=1000).fit(
        model.predict_proba(Xall)[:, 1].reshape(-1, 1), deals["y"].to_numpy())
    bundle = {"model": model, "platt": cal, "feature_columns": feats,
              "bundle_format": "dealwin-lgbm-v2", "grain": "deal",
              "label": "StageResolved=Executed (won) vs Cancelled/Dead (lost), CreatedAt>=2024",
              "group_key": "TenantId", "excluded_leaky": ["LeadSource/leadBroker", "EconArea", "EconStartRent"],
              "calibration": {"method": "platt", "prob_floor": 0.02, "prob_ceil": 0.98},
              "metrics": results, "ablation_zi_helps": bool(zi_helps), "graduates": graduates,
              "version": VERSION, "trained_at": TS, "algorithm": "lightgbm(shallow)+platt",
              "status": "shadow", "source": "rea.YardiDeal resolved 2024+; zi.Company + zi.Scoop (PIT)"}
    # cohort_rate needs to be recomputable at serve time; persist the band map + global.
    bundle["cohort_rate_map"] = {b: cr_all(b) for b in ["micro", "small", "mid", "large", "unknown"]}
    d = BUNDLES / "signal_desk_dealwin" / VERSION
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, d / "bundle.joblib")
    (d / "manifest.json").write_text(json.dumps(
        {"contract": "signal-desk-dealwin", "version": VERSION, "grain": "deal",
         "status": "shadow", "graduates": graduates, "ablation_zi_helps": bool(zi_helps),
         "resolved_deals": int(len(deals)), "won": int(deals["y"].sum()),
         "win_rate": round(float(deals["y"].mean()), 3),
         "distinct_winning_tenants": int(deals.loc[deals["y"] == 1, "group"].nunique()),
         "metrics": results,
         "note": "Leakage-free rebuild (no leadBroker/Econ*); group-disjoint by TenantId; "
                 "PIT scoops; Yardi-vs-ZI ablation. SHADOW pending graduation."},
        indent=2, default=str))
    print(f"\nSaved {VERSION} (SHADOW, not promoted). best_featureset={best}")


if __name__ == "__main__":
    main()
