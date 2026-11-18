"""EXPERIMENT — can we raise the space-demand PR-AUC? Features vs LGBM tuning vs other models.

Everything runs on the SAME 180d-embargo rolling-origin folds and the SAME eval population as
the prototype, so pooled-OOF PR-AUC is directly comparable (base rate held fixed). Tests, in
order of expected payoff:
  1. FEATURES: add the untapped sources the prototype ignored -> scoop TOPICS (287-topic tags),
     INDUSTRY (categorical), funding recency + total funding, domain rank, 30d/730d windows,
     days-since-last-relocation.
  2. LGBM TUNING: current params vs a tuned set vs scale_pos_weight (imbalance).
  3. OTHER MODELS: HistGradientBoosting (native categoricals), logistic, and a tree+logistic ensemble.

Read-only DB; writes nothing (pure comparison). Run: PYTHONPATH=. .venv/bin/python scripts/experiment_spacedemand_tuning.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core import db                                          # noqa: E402

RELO = 8; HORIZON = 180
RNG = np.random.default_rng(11)
T_HIRING = {11, 12}; T_FUND_MNA = {2, 4}; T_EXECMOVE = {18, 19}
T_PAIN = {20}; T_PROJECT = {21}; T_EARN = {1}; T_AWARD = {6}; T_LAYOFF = {17}

BASE = ["cnt_90d", "cnt_180d", "cnt_365d", "hiring_365d", "fundmna_365d", "execmove_365d",
        "pain_365d", "project_365d", "earn_365d", "award_365d", "layoff_365d",
        "prior_reloc_ct", "had_prior_reloc", "days_since_scoop", "days_since_growth",
        "momentum_180d", "emp_log", "rev_log", "company_age"]
RICH = ["cnt_30d", "cnt_730d", "days_since_reloc", "topics_365d",
        "days_since_funding", "total_funding_log", "domain_rank"]
CAT = "industry_code"


def load():
    s = db.query(f"SELECT s.CompanyID, s.OriginalPublishedDate dt, m.TypeID "
                 f"FROM [{db.DB_MAIN}].zi.Scoop s LEFT JOIN [{db.DB_MAIN}].zi.ScoopTypeMap m ON m.ScoopID=s.ScoopID "
                 f"WHERE s.OriginalPublishedDate IS NOT NULL AND s.CompanyID IS NOT NULL")
    s["dt"] = pd.to_datetime(s["dt"]); s["CompanyID"] = s["CompanyID"].astype("int64")
    s["TypeID"] = pd.to_numeric(s["TypeID"], errors="coerce")
    tp = db.query(f"SELECT s.CompanyID, s.OriginalPublishedDate dt FROM [{db.DB_MAIN}].zi.Scoop s "
                  f"JOIN [{db.DB_MAIN}].zi.ScoopTopicMap m ON m.ScoopID=s.ScoopID WHERE s.OriginalPublishedDate IS NOT NULL")
    tp["dt"] = pd.to_datetime(tp["dt"]); tp["CompanyID"] = tp["CompanyID"].astype("int64")
    c = db.query(f"SELECT CompanyID, State, EmployeeCount, Revenue, FoundedYear, DomainRank, "
                 f"TotalFundingAmount, RecentFundingDate FROM [{db.DB_MAIN}].zi.Company")
    c["CompanyID"] = c["CompanyID"].astype("int64"); c["RecentFundingDate"] = pd.to_datetime(c["RecentFundingDate"], errors="coerce")
    ind = db.query(f"SELECT ci.CompanyID, i.Name FROM [{db.DB_MAIN}].zi.CompanyIndustry ci "
                   f"JOIN [{db.DB_MAIN}].zi.Industry i ON i.IndustryID=ci.IndustryID "
                   f"WHERE ci.PrimaryRank=(SELECT MIN(PrimaryRank) FROM [{db.DB_MAIN}].zi.CompanyIndustry x WHERE x.CompanyID=ci.CompanyID)")
    ind["CompanyID"] = ind["CompanyID"].astype("int64")
    icodes = {n: i for i, n in enumerate(sorted(ind["Name"].dropna().unique()))}
    ind_map = {int(r.CompanyID): icodes.get(r.Name, -1) for r in ind.itertuples(index=False)}
    return s, tp, c, ind_map


def build(s, tp, c, ind_map):
    firmo = {int(r.CompanyID): r for r in c.itertuples(index=False)}
    topics_by = {cid: g["dt"].values.astype("datetime64[ns]") for cid, g in tp.groupby("CompanyID")}
    asofs = pd.date_range("2022-07-01", "2026-01-01", freq="MS")   # labelable only (faster)
    label_max = pd.Timestamp("2026-06-30"); D = np.timedelta64(1, "D")
    rows = []
    for cid, g in s.groupby("CompanyID"):
        g = g.sort_values("dt"); dts = g["dt"].values.astype("datetime64[ns]"); typ = g["TypeID"].to_numpy()
        first = pd.Timestamp(dts.min()); fm = firmo.get(int(cid))
        emp = getattr(fm, "EmployeeCount", None) if fm else None; rev = getattr(fm, "Revenue", None) if fm else None
        fy = getattr(fm, "FoundedYear", None) if fm else None; state = getattr(fm, "State", None) if fm else None
        dr = getattr(fm, "DomainRank", None) if fm else None; tf = getattr(fm, "TotalFundingAmount", None) if fm else None
        rfd = getattr(fm, "RecentFundingDate", None) if fm else None
        tdts = topics_by.get(int(cid))
        for a in asofs:
            if a < first:
                continue
            an = np.datetime64(a); prior = dts <= an
            if not prior.any():
                continue
            pdts, ptyp = dts[prior], typ[prior]
            w30, w90, w180, w365, w730 = (pdts > an - k * D for k in (30, 90, 180, 365, 730))
            wprev180 = (pdts <= an - 180 * D) & (pdts > an - 360 * D)
            gmask = np.isin(ptyp, list(T_HIRING | T_FUND_MNA)); rmask = ptyp == RELO
            fut = (dts > an) & (dts <= an + HORIZON * D)
            y = int(np.isin(typ[fut], [RELO]).any()) if (a + pd.Timedelta(days=HORIZON)) <= label_max else -1
            tc365 = int(((tdts > an - 365 * D) & (tdts <= an)).sum()) if tdts is not None else 0
            rows.append({
                "CompanyID": int(cid), "as_of": a, "state": state, "y": y,
                "cnt_30d": float(np.log1p(w30.sum())), "cnt_90d": float(np.log1p(w90.sum())),
                "cnt_180d": float(np.log1p(w180.sum())), "cnt_365d": float(np.log1p(w365.sum())),
                "cnt_730d": float(np.log1p(w730.sum())),
                "hiring_365d": float(np.log1p(np.isin(ptyp[w365], list(T_HIRING)).sum())),
                "fundmna_365d": float(np.isin(ptyp[w365], list(T_FUND_MNA)).sum()),
                "execmove_365d": float(np.isin(ptyp[w365], list(T_EXECMOVE)).sum()),
                "pain_365d": float(np.isin(ptyp[w365], list(T_PAIN)).sum()),
                "project_365d": float(np.log1p(np.isin(ptyp[w365], list(T_PROJECT)).sum())),
                "earn_365d": float(np.isin(ptyp[w365], list(T_EARN)).sum()),
                "award_365d": float(np.isin(ptyp[w365], list(T_AWARD)).sum()),
                "layoff_365d": float(np.isin(ptyp[w365], list(T_LAYOFF)).sum()),
                "prior_reloc_ct": float(rmask.sum()), "had_prior_reloc": float(rmask.any()),
                "days_since_scoop": float((an - pdts.max()) / D),
                "days_since_growth": float((an - pdts[gmask].max()) / D) if gmask.any() else 9999.0,
                "days_since_reloc": float((an - pdts[rmask].max()) / D) if rmask.any() else 9999.0,
                "momentum_180d": float(w180.sum() - wprev180.sum()),
                "topics_365d": float(np.log1p(tc365)),
                "emp_log": float(np.log1p(emp)) if (emp and emp > 0) else np.nan,
                "rev_log": float(np.log1p(rev)) if (rev and rev > 0) else np.nan,
                "company_age": float(a.year - fy) if (fy and fy > 1800) else np.nan,
                "domain_rank": float(dr) if (dr and dr > 0) else np.nan,
                "total_funding_log": float(np.log1p(float(tf))) if (tf and float(tf) > 0) else 0.0,
                "days_since_funding": float((an - np.datetime64(rfd)) / D) if (rfd is not None and pd.notna(rfd)) else 9999.0,
                "industry_code": int(ind_map.get(int(cid), -1)),
            })
    return pd.DataFrame(rows)


STARTS = pd.date_range("2024-07-01", "2026-01-01", freq="6MS")


def wf(lab, feats, make, cat=False, spw=None):
    oy, op = [], []
    for s in STARTS:
        te_end = s + pd.DateOffset(months=6)
        tr = lab[lab["as_of"] <= (s - pd.Timedelta(days=HORIZON))]
        te = lab[(lab["as_of"] >= s) & (lab["as_of"] < te_end)]
        if len(tr) < 500 or tr["y"].sum() < 20 or te["y"].sum() < 10:
            continue
        Xtr, Xte = tr[feats].copy(), te[feats].copy()
        if cat:
            Xtr[CAT] = Xtr[CAT].astype("category"); Xte[CAT] = Xte[CAT].astype("category")
        m = make(spw) if spw is not None else make()
        m.fit(Xtr, tr["y"].to_numpy())
        op += list(m.predict_proba(Xte)[:, 1]); oy += list(te["y"].to_numpy())
    return np.array(oy), np.array(op)


def lift_ci(y, p, n=1500):
    d = []
    for _ in range(n):
        idx = RNG.integers(0, len(y), len(y))
        if y[idx].sum() < 2:
            continue
        d.append(average_precision_score(y[idx], p[idx]) - y[idx].mean())
    return round(float(np.mean(d)), 4), round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)


def lgbm(spw=None):
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=16, max_depth=4,
                              min_child_samples=50, subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0,
                              reg_lambda=2.0, random_state=7, n_jobs=2, verbose=-1, scale_pos_weight=spw)


def lgbm_tuned(spw=None):
    return lgb.LGBMClassifier(n_estimators=700, learning_rate=0.02, num_leaves=31, max_depth=5,
                              min_child_samples=30, subsample=0.8, colsample_bytree=0.7, reg_alpha=0.5,
                              reg_lambda=1.0, random_state=7, n_jobs=2, verbose=-1, scale_pos_weight=spw)


def histgb(spw=None):
    return HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=400,
                                          l2_regularization=1.0, categorical_features="from_dtype",
                                          random_state=7)


def logit(spw=None):
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                     ("lr", LogisticRegression(C=0.5, max_iter=3000, random_state=7,
                                               class_weight=("balanced" if spw else None)))])


def main():
    print("Loading (read-only)..."); s, tp, c, ind_map = load()
    print(f"  scoops={len(s)} topic_tags={len(tp)} industries={len(set(ind_map.values()))}")
    print("Building augmented panel..."); panel = build(s, tp, c, ind_map)
    lab = panel[panel["y"] >= 0].copy()
    base = float(lab["y"].mean())
    spw = (1 - base) / base
    print(f"  labelable rows={len(lab)} pos={int(lab['y'].sum())} base={base:.4f} scale_pos_weight={spw:.1f}\n")

    NUM = BASE + RICH
    configs = [
        ("1 base LGBM (19 feats, ref)",        BASE,         lgbm,       False, None),
        ("2 rich LGBM (+topics/ind/fund)",     NUM + [CAT],  lgbm,       True,  None),
        ("3 rich LGBM + scale_pos_weight",     NUM + [CAT],  lgbm,       True,  spw),
        ("4 rich LGBM tuned",                  NUM + [CAT],  lgbm_tuned, True,  None),
        ("5 rich HistGradientBoosting",        NUM + [CAT],  histgb,     True,  None),
        ("6 rich logistic (no ind cat)",       NUM,          logit,      False, None),
    ]
    store = {}
    print(f"{'config':<36}{'PR-AUC':>8}{'xbase':>7}{'ROC':>7}")
    for name, feats, make, cat, sp in configs:
        y, p = wf(lab, feats, make, cat=cat, spw=sp)
        pr = average_precision_score(y, p); roc = roc_auc_score(y, p)
        store[name] = (y, p, pr)
        print(f"{name:<36}{pr:>8.4f}{pr/base:>6.1f}x{roc:>7.3f}")

    # ensemble: mean of best tree (tuned) + logistic, same folds
    yb, pb, _ = store["4 rich LGBM tuned"]; yl, pl, _ = store["6 rich logistic (no ind cat)"]
    if np.array_equal(yb, yl):
        pe = (pb + pl) / 2; pre = average_precision_score(yb, pe)
        store["7 ensemble (tuned+logit)"] = (yb, pe, pre)
        print(f"{'7 ensemble (tuned LGBM + logistic)':<36}{pre:>8.4f}{pre/base:>6.1f}x{roc_auc_score(yb,pe):>7.3f}")

    best = max(store, key=lambda k: store[k][2])
    ref = store['1 base LGBM (19 feats, ref)'][2]
    yb, pb, prb = store[best]
    lm, lo, hi = lift_ci(yb, pb)
    print(f"\nBEST: {best}  PR-AUC={prb:.4f} (ref base_lgbm={ref:.4f}, delta {prb-ref:+.4f})")
    print(f"  lift vs no-skill mean={lm} CI95=[{lo},{hi}]")
    print("\nTakeaway: compare col 1 (features off) vs 2 (features on) = FEATURE gain; "
          "2 vs 3/4 = TUNING gain; 4 vs 5/6/7 = MODEL choice.")


if __name__ == "__main__":
    main()
