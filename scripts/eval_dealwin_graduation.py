"""Deal-win model graduation checks (per docs/scoring-and-win-model.md gate).
1. leadBroker ablation: does the AUC edge over the cohort baseline survive removing leadBroker?
2. Beat live P(win) (cohort x concept-lift) -- best-effort using CURRENT tags (LEAKY; indicative).
3. Temporal-split integrity: are test-era deals reactivated old rows / bulk-ModifiedAt?
Read-only. Reuses scripts/train_brokerage_dealwin.py helpers.
Run: PYTHONPATH=. .venv/bin/python scripts/eval_dealwin_graduation.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from core import db                       # noqa: E402
import train_brokerage_dealwin as dw      # noqa: E402

RNG = np.random.default_rng(12345)
LIFT = {"has-active-deal": 1.5, "lease-expiring-12mo": 1.3, "expanding-footprint": 1.25,
        "growth-mode": 1.2, "has-recent-scoop": 1.15, "in-pipeline": 1.1, "distress": 0.95,
        "downsizing": 0.85, "stale": 0.85, "cold": 0.8}


def boot_diff(y, a, b, n=2000):
    d = []
    for _ in range(n):
        idx = RNG.integers(0, len(y), len(y))
        if y[idx].sum() < 2 or y[idx].sum() == len(idx):
            continue
        d.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    return round(float(np.mean(d)), 3), [round(float(np.percentile(d, 2.5)), 3), round(float(np.percentile(d, 97.5)), 3)]


def main():
    deals = dw.load_deals()
    try:
        ck = db.query(f"SELECT CompanyID, ConceptKeys FROM [{db.DB_MAIN}].rea.CompanyFeatureSnapshot WHERE ConceptKeys IS NOT NULL")
        ckmap = {int(r.CompanyID): str(r.ConceptKeys) for r in ck.itertuples(index=False)}
    except Exception as e:
        ckmap = {}; print("concept tags unavailable:", str(e)[:120])

    for sp in dw.SPLITS:
        tr, te = deals[deals.ModifiedAt < sp], deals[deals.ModifiedAt >= sp]
        if len(te) < 20 or te.y.sum() < 5:
            continue
        cr, _ = dw.cohort_fn(tr)
        Xtr = dw.base_matrix(tr, cr)[dw.BASE_FEATS]; ytr = tr.y.to_numpy()
        Xte = dw.base_matrix(te, cr)[dw.BASE_FEATS]; yte = te.y.to_numpy()
        pfull = dw.fit_lr(Xtr, ytr).predict_proba(Xte.values)[:, 1]
        abl = [f for f in dw.BASE_FEATS if f != "leadBroker"]
        pabl = dw.fit_lr(Xtr[abl], ytr).predict_proba(Xte[abl].values)[:, 1]
        base = np.array([cr(b) for b in te.band])
        lift = []
        for r in te.itertuples(index=False):
            l = 1.0
            keys = ckmap.get(int(r.zi)) if pd.notna(r.zi) else None
            if keys:
                for k in str(keys).split(","):
                    if k.strip() in LIFT:
                        l *= LIFT[k.strip()]
            lift.append(min(2.0, max(0.5, l)))
        plive = np.clip(base * np.array(lift), 0.02, 0.85)

        d_fb, ci_fb = boot_diff(yte, pfull, base)
        d_ab, ci_ab = boot_diff(yte, pabl, base)
        d_fl, ci_fl = boot_diff(yte, pfull, plive)
        te_pre = int((te.CreatedAt < sp).sum())
        topdate = int(te.ModifiedAt.dt.date.value_counts().iloc[0]) if len(te) else 0
        print(f"\n=== split {sp}: test n={len(te)} win={yte.mean():.2f} ===")
        print(f"  AUC: full {roc_auc_score(yte,pfull):.3f} | no-leadBroker {roc_auc_score(yte,pabl):.3f} | "
              f"cohort {roc_auc_score(yte,base):.3f} | live P(win)~ {roc_auc_score(yte,plive):.3f}")
        print(f"  full vs cohort:       {d_fb} CI{ci_fb}")
        print(f"  ABLATED vs cohort:    {d_ab} CI{ci_ab}   <- edge survives leadBroker removal if CI>0")
        print(f"  full vs live P(win):  {d_fl} CI{ci_fl}   <- (live P(win) uses CURRENT tags = leaky/indicative)")
        print(f"  leadBroker prevalence: train {tr.leadBroker.mean():.3f}  test {te.leadBroker.mean():.3f}")
        print(f"  temporal: {te_pre}/{len(te)} test deals created BEFORE split (long deals, expected); "
              f"max single ModifiedAt-date count = {topdate}")
    print("\nVERDICT inputs: ablated-vs-cohort CI>0 on the 2 larger splits => edge is not just leadBroker.")
    print("NOTE: live P(win) baseline uses current concept tags (no point-in-time tag history) -> indicative only.")


if __name__ == "__main__":
    main()
