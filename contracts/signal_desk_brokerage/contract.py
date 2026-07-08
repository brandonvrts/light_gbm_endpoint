"""Brokerage scoring: resolve a company -> pull its full snapshot history from SSMS ->
run the model across all of them (the historical-prediction series)."""
from __future__ import annotations
import numpy as np

from core import registry
from contracts.signal_desk_brokerage import data, prepare_features as prep, scorecard


def _bundle() -> dict:
    return registry.load("signal_desk_brokerage")


def _tier(p: float, th: dict) -> str:
    return "low" if p < th["risk_low_max"] else ("medium" if p < th["risk_medium_max"] else "high")


def score_company(query: str, include_history: bool = True) -> dict:
    matches = data.resolve_company(query)
    if not matches:
        return {"contract": "signal-desk-brokerage", "matches": [], "error": f"no company for '{query}'"}
    co = matches[0]
    hmy = int(co["hMy"])
    b = _bundle()
    history = []
    hist_df = data.feature_history(hmy)
    if not hist_df.empty:
        X = prep.prepare(hist_df, b["feature_medians"])
        feats = b.get("feature_columns") or list(X.columns)   # model may use a subset (e.g. scoop-free)
        p = np.clip(b["model"].predict_proba(X[feats])[:, 1],
                    b["calibration"]["prob_floor"], b["calibration"]["prob_ceil"])
        th = b["thresholds"]
        hd = hist_df.reset_index(drop=True)
        for i in range(len(hd)):
            history.append({"snapshot_date": hd.iloc[i]["SnapshotDate"],
                            "conversion_probability": round(float(p[i]), 4),
                            "risk_tier": _tier(float(p[i]), th)})
    rules = data.current_rules_score(hmy)
    rules_stale = False
    if rules and rules.get("ScoredAt"):
        try:
            from datetime import date as _d
            sa = rules["ScoredAt"]
            sa = sa.date() if hasattr(sa, "date") else _d.fromisoformat(str(sa)[:10])
            rules_stale = (_d.today() - sa).days > 35
        except Exception:
            pass
    return {"contract": "signal-desk-brokerage", "model_version": b.get("version"),
            "model_source": "standin" if b.get("bundle_format") == "standin" else (b.get("source") or "real-production"),
            "company": {"hMy": hmy, "display_name": co.get("display_name"),
                        "market": co.get("market"), "industry": co.get("zi_primary_industry")},
            "candidates": matches if len(matches) > 1 else [],
            "scorecard": scorecard.scorecard_for_prospect(hmy),  # company-centric headline (via ProspectCompanyLink)
            "current_rules_score": rules,
            "rules_score_stale": rules_stale,
            "history_count": len(history),
            "data_as_of": history[-1]["snapshot_date"] if history else None,
            "history": history if include_history else history[-1:]}


def list_companies(limit: int = 50) -> dict:
    return {"contract": "signal-desk-brokerage", "companies": data.list_companies(limit)}
