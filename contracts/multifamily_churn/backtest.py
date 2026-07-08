"""Multifamily backtest / out-of-time validation.

Score the active-resident cohort as of a HISTORICAL date and compare the model's
churn_probability against the realized ActualChurn (returned by the feature SP) ->
ROC-AUC, Brier, log-loss, base rate, and a calibration table. Because the label is
'moved out within the 180d horizon', the as-of must be >= ~180 days ago or the
outcome is still censored (flagged).

Uses the same scoring path as production (identical prep), sampled via `limit` so a
run returns quickly, and cached by (property_ids, as_of, limit) via score_buildings.
"""
from __future__ import annotations
from datetime import date

import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

from contracts.multifamily_churn import contract as mf

HORIZON = 180


def backtest(property_ids: list[int], as_of: str | None = None, limit: int | None = 250) -> dict:
    res = mf.score_buildings(property_ids or [62, 72, 82, 26], as_of=as_of, limit=limit)
    rows = res.get("results", [])
    y, p = [], []
    for r in rows:
        a = r.get("actual_churn")
        try:
            ai = int(a)
        except (TypeError, ValueError):
            continue
        if ai in (0, 1):
            y.append(ai)
            p.append(float(r.get("churn_probability") or 0.0))
    y_arr, p_arr = np.array(y), np.array(p)
    out = {
        "contract": "multifamily-churn", "view": "backtest", "as_of": res.get("as_of"),
        "model_version": res.get("model_version"), "property_ids": res.get("property_ids"),
        "eligible_units": res.get("eligible_units"), "scored_units": res.get("scored_units"),
        "labeled_units": int(len(y_arr)), "horizon_days": HORIZON,
    }
    try:
        aod = date.fromisoformat(str(res.get("as_of"))[:10])
        out["label_censored"] = (date.today() - aod).days < HORIZON
    except Exception:
        out["label_censored"] = None

    if len(y_arr) < 10 or len(set(y_arr.tolist())) < 2:
        out["error"] = ("Too few labeled units or a single outcome class — pick an older as-of "
                        "date (>= 180 days ago) with churn variation.")
        return out

    base = float(y_arr.mean())
    pc = np.clip(p_arr, 1e-6, 1 - 1e-6)
    out.update({
        "base_rate": round(base, 4),
        "roc_auc": round(float(roc_auc_score(y_arr, p_arr)), 4),
        "brier": round(float(brier_score_loss(y_arr, p_arr)), 4),
        "log_loss": round(float(log_loss(y_arr, pc, labels=[0, 1])), 4),
        "mean_predicted": round(float(p_arr.mean()), 4),
    })
    edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    cal = []
    for i in range(5):
        lo, hi = edges[i], edges[i + 1]
        m = (p_arr >= lo) & (p_arr <= hi if i == 4 else p_arr < hi)
        if int(m.sum()):
            cal.append({"bin": f"{int(lo * 100)}-{int(hi * 100)}%", "n": int(m.sum()),
                        "pred_mean": round(float(p_arr[m].mean()), 3),
                        "actual_rate": round(float(y_arr[m].mean()), 3)})
    out["calibration"] = cal
    out["walkforward_auc_ref"] = 0.893
    out["caveat"] = ("ROC-AUC on this cohort can run HIGHER than the leakage-audited walk-forward "
                     "(0.893) because move-out-notice features are near-deterministic for late-stage "
                     "leases, and the active-resident cohort differs from the anchor-snapshot validation "
                     "design. Treat the CALIBRATION table (predicted vs actual per band) as the more "
                     "robust read; ROC here is a directional out-of-time check, not the headline metric.")
    out["note"] = ("Out-of-time validation: churn_probability vs realized ActualChurn on the as-of "
                   "cohort (sampled by `limit`). Use an as-of >= 180d ago so outcomes are realized.")
    return out
