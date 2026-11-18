"""Duration-model test battery — the 'thoroughly test it' pass.

What it probes (each targets a specific way v0.1's single 2022 exam could mislead):
  1. MULTI-YEAR FROZEN EXAMS — one frozen prediction per anchor (2022-01 .. 2025-07),
     each trained only on anchors ending >=6 months earlier. Evaluation caps adapt to
     what is actually observable from each anchor (no censored labels, ever).
  2. AGING-MODEL TEST — anchors 2023+ are scored by a model whose training data STOPS
     at 2022-07 (the last anchor whose 48-month labels are fully resolved). That is
     precisely the production situation: an old model scoring a new era.
  3. MEMORIZATION SPLIT — every frozen exam is reported for residents the model SAW at
     earlier training anchors vs residents it has NEVER seen. If skill collapses on the
     never-seen group, the headline was person-memorization.
  4. CALIBRATION DRIFT — predicted vs realized mean months per anchor (level bias by era).
  5. EDGE-V2 STABILITY — month-over-month membership overlap of the fight-for list
     (churn p>=0.2, no notice, top half by trait score) across recent snapshots.

Run: PYTHONPATH=. .venv/bin/python scripts/backtest_multifamily_duration.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

import importlib.util
_spec = importlib.util.spec_from_file_location("tmd", ROOT / "scripts" / "train_multifamily_duration.py")
tmd = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(tmd)

TODAY = pd.Timestamp("2026-07-01")           # last fully-elapsed month boundary
POOL_ANCHORS = [f"{y}-{m:02d}-01" for y in range(2016, 2023) for m in (1, 7)]  # ..2022-07
EVAL_ANCHORS = ["2022-01-01", "2022-07-01", "2023-01-01", "2023-07-01",
                "2024-01-01", "2024-07-01", "2025-01-01", "2025-07-01"]
CAP = 48.0
OUT_JSON = ROOT.parent / "outputs" / "multifamily-prediction" / "reports" / \
    "duration-battery-2026-07-22.json"


def observable_months(anchor: str) -> float:
    return (TODAY - pd.Timestamp(anchor)).days / 30.44


def fit(train: pd.DataFrame, feats: list[str]):
    m = LGBMRegressor(objective="l1", n_estimators=700, learning_rate=0.05,
                      num_leaves=31, min_child_samples=30, subsample=0.9,
                      colsample_bytree=0.9, random_state=7, verbose=-1)
    m.fit(tmd._matrix(train, feats), train["remaining_months"].to_numpy())
    return m


def evaluate(pred: np.ndarray, real: np.ndarray, cap: float, obs: float) -> dict:
    p = np.clip(pred, 0, cap); r = np.clip(real, 0, cap)
    out = {"n": int(len(r)), "eval_cap": round(cap, 1),
           "mae": round(float(np.mean(np.abs(p - r))), 2),
           "spearman": round(float(spearmanr(p, r).statistic), 3),
           "pred_mean": round(float(p.mean()), 1), "real_mean": round(float(r.mean()), 1)}
    for hz in (12, 24):
        if obs >= hz + 0.5:
            y = (r < hz).astype(int)
            if 0 < y.sum() < len(y):
                out[f"auc_leave{hz}"] = round(float(roc_auc_score(y, -p)), 3)
    return out


def main() -> None:
    print("lineage ends...")
    ends = tmd._lineage_ends()
    panels: dict[str, pd.DataFrame] = {}
    for a in sorted(set(POOL_ANCHORS + EVAL_ANCHORS)):
        panels[a] = tmd._panel(a, ends)
        print(f"  {a}: {len(panels[a])}")

    results = []
    for ev in EVAL_ANCHORS:
        cutoff = pd.Timestamp(ev) - pd.DateOffset(months=6)
        train_anchors = [a for a in POOL_ANCHORS if pd.Timestamp(a) <= cutoff]
        train = pd.concat([panels[a] for a in train_anchors], ignore_index=True)
        bt = panels[ev]
        obs = observable_months(ev)
        cap = min(CAP, np.floor(obs))
        seen = set(train["lineage_terminal"].astype(int))
        is_seen = bt["lineage_terminal"].astype(int).isin(seen).to_numpy()
        row = {"anchor": ev, "train_end": train_anchors[-1], "train_rows": int(len(train)),
               "observable_months": round(obs, 1),
               "seen_share": round(float(is_seen.mean()), 2)}
        for tag, feats in tmd.VARIANTS.items():
            model = fit(train, feats)
            pred = model.predict(tmd._matrix(bt, feats))
            real = bt["remaining_months"].to_numpy()
            row[tag] = evaluate(pred, real, cap, obs)
            row[tag]["seen"] = evaluate(pred[is_seen], real[is_seen], cap, obs)
            row[tag]["unseen"] = (evaluate(pred[~is_seen], real[~is_seen], cap, obs)
                                  if (~is_seen).sum() >= 25 else {"n": int((~is_seen).sum())})
        # tenure baseline (rank + classification, same caps)
        ten = pd.to_numeric(bt["TenureMonths"], errors="coerce").fillna(0).to_numpy()
        r = np.clip(bt["remaining_months"].to_numpy(), 0, cap)
        row["baseline_tenure_spearman"] = round(float(spearmanr(ten, r).statistic), 3)
        if obs >= 12.5:
            y = (r < 12).astype(int)
            if 0 < y.sum() < len(y):
                row["baseline_tenure_auc12"] = round(float(roc_auc_score(y, -ten)), 3)
        results.append(row)
        t = row["trait"]
        print(f"{ev}  trait: rho={t['spearman']} mae={t['mae']} "
              f"auc12={t.get('auc_leave12', '—')} | unseen(n={t['unseen'].get('n')}): "
              f"rho={t['unseen'].get('spearman', '—')} auc12={t['unseen'].get('auc_leave12', '—')} "
              f"| level {t['pred_mean']} vs {t['real_mean']}")

    # ---- edge-v2 month-over-month stability (current production bundle) ----
    import joblib
    from contracts.multifamily_churn import cohorts
    B = joblib.load(ROOT / "bundles" / "multifamily_duration" / "v0.1-jan22-backtest" / "model.joblib")
    v = B["variants"]["trait"]
    rosters = {}
    for asof in ("2026-05-01", "2026-06-01", "2026-07-01"):
        df, _b = cohorts._scored_frame(None, asof)
        X = pd.DataFrame({c: pd.to_numeric(df.get(c), errors="coerce") for c in v["features"]})
        df = df.copy(); df["trait"] = np.clip(v["model"].predict(X), 0, B["cap_months"])
        nn = df[pd.to_numeric(df.get("HasNotice"), errors="coerce").fillna(0) == 0]
        risk = nn[nn["p_churn"] >= 0.20]
        med = float(risk["trait"].median()) if len(risk) else 0.0
        rosters[asof] = set(risk[risk["trait"] >= med]["SCODE"].astype(str).str.strip())
        print(f"edge-v2 @ {asof}: n={len(rosters[asof])}")
    months = list(rosters)
    stab = []
    for a, b in zip(months, months[1:]):
        inter = len(rosters[a] & rosters[b]); union = len(rosters[a] | rosters[b])
        stab.append({"from": a, "to": b, "overlap": inter,
                     "jaccard": round(inter / union, 2) if union else None})
        print(f"stability {a} -> {b}: overlap={inter} jaccard={stab[-1]['jaccard']}")

    OUT_JSON.write_text(json.dumps({"frozen_exams": results, "edge_stability": stab}, indent=1))
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
