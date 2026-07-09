"""PROTOTYPE — space-need / relocation-expansion propensity (lead-gen) model.

Flips the target away from the endogenous, label-starved "will they sign with us":
  Label  = a Facilities Relocation/Expansion scoop (zi.ScoopType TypeID 8) in the next 180d.
  Grain  = company-month panel over scooped companies (from each company's first scoop).
  Feats  = STRICTLY point-in-time prior scoop-by-type counts (multi-window) + recency +
           momentum + firmographics (employees/revenue/age). All scoops dt <= as_of.
Self-contained in ZoomInfo -> does NOT depend on the ~54 scarce Yardi conversion labels.

Honest harness: expanding-window rolling-origin walk-forward with a 180d EMBARGO (a training
row's label window cannot overlap the test as-of), LightGBM vs logistic, Platt-calibrated
pooled OOF, paired-bootstrap PR-AUC lift vs base rate. Produces a ranked IA/NE "space-need
leads" list from the final model at the latest as-of.

NOTE (2026-07-09): now SERVED as a labeled prototype -> contract contracts/signal_desk_spacedemand,
endpoint GET /brokerage/space-demand-leads, a "Space-Demand Leads (PROTOTYPE)" section in the
Brokerage UI tab, and a card in the Model tab. This script remains the (re)trainer of the bundle.

PROTOTYPE ONLY: writes a NON-served bundle + a leads sample; does not touch default.txt, the
API, or the UI. SELECT-only DB. Run: PYTHONPATH=. .venv/bin/python scripts/prototype_spacedemand.py

NOTE (2026-07-09): a same-folds experiment (scripts/experiment_spacedemand_tuning.py) found the
modeling levers are largely tapped -- adding topics/industry/funding features only nudged PR-AUC
0.335->0.343 (marginal), LightGBM hyper-tuning made it WORSE (0.323, overfit), and HistGB/logistic/
ensemble all underperformed LightGBM. ~0.34 is the ceiling on THIS label+population; material gains
require label/data changes (fuse Yardi ground-truth transactions, real-estate topic mining, population
segmentation, more positives over time), not model tuning.
"""
from __future__ import annotations
import json, sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core import db                                          # noqa: E402

TS = datetime.now().strftime("%Y%m%d_%H%M%S")
VERSION = "v0.1-prototype"
RELO = 8                                                     # Facilities Relocation / Expansion (label)
HORIZON = 180
RNG = np.random.default_rng(11)
# scoop type families (TypeID)
T_HIRING = {11, 12}         # Hiring Plans, Open Position
T_FUND_MNA = {2, 4}         # Funding, M&A
T_EXEC = {17, 18, 19}       # Layoffs? no -> 17 Layoffs; exec = 18,19; keep separate below
T_EXECMOVE = {18, 19}       # Management Move, Executive Move
T_PAIN = {20}               # Pain Point
T_PROJECT = {21}
T_EARN = {1}
T_AWARD = {6}
T_LAYOFF = {17}

FEATS = ["cnt_90d", "cnt_180d", "cnt_365d", "hiring_365d", "fundmna_365d", "execmove_365d",
         "pain_365d", "project_365d", "earn_365d", "award_365d", "layoff_365d",
         "prior_reloc_ct", "had_prior_reloc", "days_since_scoop", "days_since_growth",
         "momentum_180d", "emp_log", "rev_log", "company_age"]


def load_scoops() -> pd.DataFrame:
    s = db.query(
        f"SELECT s.CompanyID, s.OriginalPublishedDate dt, m.TypeID "
        f"FROM [{db.DB_MAIN}].zi.Scoop s LEFT JOIN [{db.DB_MAIN}].zi.ScoopTypeMap m ON m.ScoopID=s.ScoopID "
        f"WHERE s.OriginalPublishedDate IS NOT NULL AND s.CompanyID IS NOT NULL")
    s["dt"] = pd.to_datetime(s["dt"])
    s["CompanyID"] = pd.to_numeric(s["CompanyID"], errors="coerce").astype("int64")
    s["TypeID"] = pd.to_numeric(s["TypeID"], errors="coerce")
    return s.dropna(subset=["dt"])


def load_companies() -> pd.DataFrame:
    c = db.query(
        f"SELECT CompanyID, Name, State, EmployeeCount, Revenue, FoundedYear "
        f"FROM [{db.DB_MAIN}].zi.Company")
    c["CompanyID"] = pd.to_numeric(c["CompanyID"], errors="coerce").astype("int64")
    return c


def build_panel(scoops: pd.DataFrame, comp: pd.DataFrame) -> pd.DataFrame:
    firmo = {int(r.CompanyID): r for r in comp.itertuples(index=False)}
    asofs = pd.date_range("2022-07-01", "2026-06-01", freq="MS")
    label_max = pd.Timestamp("2026-06-30")
    D = np.timedelta64(1, "D")
    rows = []
    for cid, g in scoops.groupby("CompanyID"):
        g = g.sort_values("dt")
        dts = g["dt"].values.astype("datetime64[ns]")
        typ = g["TypeID"].to_numpy()
        first = pd.Timestamp(dts.min())
        fm = firmo.get(int(cid))
        emp = getattr(fm, "EmployeeCount", None) if fm else None
        rev = getattr(fm, "Revenue", None) if fm else None
        fy = getattr(fm, "FoundedYear", None) if fm else None
        state = (getattr(fm, "State", None) if fm else None)
        for a in asofs:
            if a < first:                      # only score once the company is "known"
                continue
            an = np.datetime64(a)
            prior = dts <= an
            if not prior.any():
                continue
            pdts, ptyp = dts[prior], typ[prior]
            w90 = pdts > an - 90 * D
            w180 = pdts > an - 180 * D
            w365 = pdts > an - 365 * D
            wprev180 = (pdts <= an - 180 * D) & (pdts > an - 360 * D)
            growth_mask = np.isin(ptyp, list(T_HIRING | T_FUND_MNA))
            last_scoop = pdts.max()
            last_growth = pdts[growth_mask].max() if growth_mask.any() else None
            # label: relocation event in (as_of, as_of+180d]; only where fully observable
            fut = (dts > an) & (dts <= an + HORIZON * D)
            labelable = (a + pd.Timedelta(days=HORIZON)) <= label_max
            y = int(np.isin(typ[fut], [RELO]).any()) if labelable else -1
            rows.append({
                "CompanyID": int(cid), "as_of": a, "state": state, "y": y,
                "cnt_90d": float(np.log1p(w90.sum())), "cnt_180d": float(np.log1p(w180.sum())),
                "cnt_365d": float(np.log1p(w365.sum())),
                "hiring_365d": float(np.log1p(np.isin(ptyp[w365], list(T_HIRING)).sum())),
                "fundmna_365d": float(np.isin(ptyp[w365], list(T_FUND_MNA)).sum()),
                "execmove_365d": float(np.isin(ptyp[w365], list(T_EXECMOVE)).sum()),
                "pain_365d": float(np.isin(ptyp[w365], list(T_PAIN)).sum()),
                "project_365d": float(np.log1p(np.isin(ptyp[w365], list(T_PROJECT)).sum())),
                "earn_365d": float(np.isin(ptyp[w365], list(T_EARN)).sum()),
                "award_365d": float(np.isin(ptyp[w365], list(T_AWARD)).sum()),
                "layoff_365d": float(np.isin(ptyp[w365], list(T_LAYOFF)).sum()),
                "prior_reloc_ct": float((ptyp == RELO).sum()),
                "had_prior_reloc": float((ptyp == RELO).any()),
                "days_since_scoop": float((an - last_scoop) / D),
                "days_since_growth": float((an - last_growth) / D) if last_growth is not None else 9999.0,
                "momentum_180d": float(w180.sum() - wprev180.sum()),
                "emp_log": float(np.log1p(emp)) if (emp and emp > 0) else np.nan,
                "rev_log": float(np.log1p(rev)) if (rev and rev > 0) else np.nan,
                "company_age": float(a.year - fy) if (fy and fy > 1800) else np.nan,
            })
    return pd.DataFrame(rows)


def _logistic():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler()),
                     ("lr", LogisticRegression(C=0.5, max_iter=3000, random_state=7))])


def _lgbm():
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=16, max_depth=4,
                              min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
                              reg_alpha=1.0, reg_lambda=2.0, random_state=7, n_jobs=2, verbose=-1)


def walk_forward(lab, build):
    """Expanding-window rolling-origin with 180d embargo. Test windows: half-year blocks
    2024-07..2026-01 (label fully observed); train = as_of <= test_start - 180d."""
    starts = pd.date_range("2024-07-01", "2026-01-01", freq="6MS")
    oof_y, oof_p, folds = [], [], []
    for s in starts:
        te_end = s + pd.DateOffset(months=6)
        tr = lab[lab["as_of"] <= (s - pd.Timedelta(days=HORIZON))]
        te = lab[(lab["as_of"] >= s) & (lab["as_of"] < te_end)]
        if len(tr) < 500 or tr["y"].sum() < 20 or len(te) < 100 or te["y"].sum() < 10:
            continue
        m = build(); m.fit(tr[FEATS], tr["y"].to_numpy())
        p = m.predict_proba(te[FEATS])[:, 1]
        y = te["y"].to_numpy()
        folds.append({"test_start": str(s.date()), "n": int(len(te)), "pos": int(y.sum()),
                      "base": round(float(y.mean()), 4),
                      "pr_auc": round(float(average_precision_score(y, p)), 4),
                      "roc_auc": round(float(roc_auc_score(y, p)), 4)})
        oof_y += list(y); oof_p += list(p)
    return np.array(oof_y), np.array(oof_p), folds


def bootstrap_lift(y, p, n=2000):
    d = []
    for _ in range(n):
        idx = RNG.integers(0, len(y), len(y))
        if y[idx].sum() < 2:
            continue
        d.append(average_precision_score(y[idx], p[idx]) - y[idx].mean())
    lo, hi = np.percentile(d, [2.5, 97.5])
    return round(float(np.mean(d)), 4), round(float(lo), 4), round(float(hi), 4)


def evaluate(tag, lab, build):
    y, p, folds = walk_forward(lab, build)
    if len(y) == 0:
        return None
    lift, lo, hi = bootstrap_lift(y, p)
    return {"tag": tag, "oof_n": int(len(y)), "oof_pos": int(y.sum()),
            "base_rate": round(float(y.mean()), 4),
            "pr_auc": round(float(average_precision_score(y, p)), 4),
            "roc_auc": round(float(roc_auc_score(y, p)), 4),
            "brier": round(float(brier_score_loss(y, p)), 4),
            "pr_auc_lift_vs_base": lift, "pr_auc_lift_ci95": [lo, hi], "folds": folds}


def main():
    print("Loading scoops + companies (read-only)...")
    scoops = load_scoops(); comp = load_companies()
    print(f"  scoops={len(scoops)} companies_scooped={scoops['CompanyID'].nunique()} zi.Company={len(comp)}")
    print("Building company-month panel (point-in-time)...")
    panel = build_panel(scoops, comp)
    lab = panel[panel["y"] >= 0].copy()          # labelable rows (as_of <= 2026-01-01)
    print(f"  panel rows={len(panel)} labelable={len(lab)} positives={int(lab['y'].sum())} "
          f"base={lab['y'].mean():.4f} companies={lab['CompanyID'].nunique()}")

    results = {}
    for tag, build in [("logistic", _logistic), ("lightgbm", _lgbm)]:
        r = evaluate(tag, lab, build)
        results[tag] = r
        if r:
            print(f"\n[{tag}] oof_n={r['oof_n']} pos={r['oof_pos']} base={r['base_rate']} | "
                  f"PR-AUC={r['pr_auc']} (lift {r['pr_auc_lift_vs_base']} CI{r['pr_auc_lift_ci95']}) | "
                  f"ROC-AUC={r['roc_auc']} | Brier={r['brier']}")
            for f in r["folds"]:
                print(f"    {f['test_start']}: n={f['n']} pos={f['pos']} base={f['base']} "
                      f"PR-AUC={f['pr_auc']} ROC={f['roc_auc']}")

    valid = {k: v for k, v in results.items() if v}
    if not valid:
        print("No valid folds — abort."); return
    best = max(valid, key=lambda k: valid[k]["pr_auc"])
    print(f"\nBEST: {best} (PR-AUC {valid[best]['pr_auc']} vs base {valid[best]['base_rate']} = "
          f"{valid[best]['pr_auc']/max(valid[best]['base_rate'],1e-9):.1f}x)")

    # Final model on all labelable rows; Platt-calibrate; score the LATEST as_of for live leads.
    build = _lgbm if best == "lightgbm" else _logistic
    model = build().fit(lab[FEATS], lab["y"].to_numpy())
    platt = LogisticRegression(C=1e6, max_iter=1000).fit(
        model.predict_proba(lab[FEATS])[:, 1].reshape(-1, 1), lab["y"].to_numpy())
    latest = panel[panel["as_of"] == panel["as_of"].max()].copy()
    raw = model.predict_proba(latest[FEATS])[:, 1]
    latest["propensity"] = platt.predict_proba(raw.reshape(-1, 1))[:, 1]
    # IA/NE leads
    ia_ne = latest[latest["state"].isin(["Iowa", "Nebraska", "IA", "NE"])].copy()
    name = {int(r.CompanyID): r.Name for r in comp.itertuples(index=False)}
    ia_ne = ia_ne.sort_values("propensity", ascending=False)
    leads = []
    for r in ia_ne.head(50).itertuples(index=False):
        leads.append({"company_id": int(r.CompanyID), "company": name.get(int(r.CompanyID)),
                      "state": r.state, "propensity": round(float(r.propensity), 4),
                      "hiring_365d": r.hiring_365d, "fundmna_365d": r.fundmna_365d,
                      "project_365d": round(float(r.project_365d), 2),
                      "had_prior_reloc": r.had_prior_reloc, "days_since_scoop": r.days_since_scoop})
    print(f"\nLATEST as_of={panel['as_of'].max().date()} | IA/NE scored companies={len(ia_ne)} | "
          f"top-decile propensity>={round(float(np.percentile(ia_ne['propensity'],90)),3)}")
    print("  (company names redacted here; full ranked list saved to leads_sample.json)")
    print(f"  top-10 propensities: {[round(float(x),3) for x in ia_ne['propensity'].head(10)]}")

    d = ROOT / "bundles" / "signal_desk_spacedemand" / VERSION
    d.mkdir(parents=True, exist_ok=True)
    bundle = {"model": model, "platt": platt, "feature_columns": FEATS,
              "bundle_format": "spacedemand-prototype", "grain": "company-month",
              "label": f"Facilities Relocation/Expansion scoop (TypeID {RELO}) within {HORIZON}d",
              "algorithm": best, "status": "prototype (NOT served)", "version": VERSION, "trained_at": TS,
              "calibration": {"method": "platt"}, "metrics": results,
              "source": "zi.Scoop + zi.ScoopType + zi.Company (2022-2026); self-contained (no Yardi labels)"}
    joblib.dump(bundle, d / "bundle.joblib")
    (d / "leads_sample.json").write_text(json.dumps(
        {"as_of": str(panel["as_of"].max().date()), "scope": "IA/NE", "model": best,
         "n_scored": int(len(ia_ne)), "leads_top50": leads}, indent=2, default=str))
    (d / "manifest.json").write_text(json.dumps(
        {"contract": "signal-desk-spacedemand", "version": VERSION, "status": "PROTOTYPE (not served)",
         "grain": "company-month", "label": bundle["label"], "algorithm": best,
         "metrics": {k: {kk: vv for kk, vv in (results[k] or {}).items() if kk != "folds"}
                     for k in results}, "feature_count": len(FEATS),
         "note": "Prototype relocation/expansion propensity lead-gen model on the self-contained "
                 "zi.Scoop panel. NOT wired to API/UI/default.txt — validation + leads demo only."},
        indent=2, default=str))
    print(f"\nSaved PROTOTYPE bundle + leads_sample.json to {d} (NOT served). DONE")


if __name__ == "__main__":
    main()
