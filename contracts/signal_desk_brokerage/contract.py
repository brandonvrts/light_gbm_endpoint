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


def _opp_tier(p: float, policy: dict | None) -> tuple[str | None, str | None]:
    """Distribution-derived OPPORTUNITY tier (high prob = hot lead). Brokerage semantics
    are inverted vs churn -- a hot lead is good, not a risk (Phase 0.3)."""
    if not policy:
        return None, None
    labels = policy.get("labels") or {"hot": "Hot lead", "warm": "Watch", "cold": "Low priority"}
    tier = "hot" if p >= policy["hot_min"] else ("warm" if p >= policy["warm_min"] else "cold")
    return tier, labels.get(tier)


def _score_history(hist_df, b: dict) -> list[dict]:
    """Build the per-snapshot prediction series. v0.5 split bundles return a combined
    conversion probability plus the pipeline_momentum and cold_fit sub-scores and a
    distribution-based opportunity_tier; legacy single-head bundles keep risk_tier."""
    floor, ceil = b["calibration"]["prob_floor"], b["calibration"]["prob_ceil"]
    X = prep.prepare(hist_df, b["feature_medians"])
    hd = hist_df.reset_index(drop=True)
    out = []
    if b.get("bundle_format") == "brokerage-split-v0.5":
        models, feats = b["models"], b["features"]
        prob = {h: np.clip(models[h].predict_proba(X[feats[h]].values)[:, 1], floor, ceil)
                for h in models}
        tiers = b.get("opportunity_tiers") or {}
        for i in range(len(hd)):
            c = float(prob["combined"][i])
            tier, label = _opp_tier(c, tiers.get("combined"))
            cf_tier, _ = _opp_tier(float(prob["cold_fit"][i]), tiers.get("cold_fit"))
            out.append({"snapshot_date": hd.iloc[i]["SnapshotDate"],
                        "conversion_probability": round(c, 4),
                        "pipeline_momentum": round(float(prob["pipeline_momentum"][i]), 4),
                        "cold_fit": round(float(prob["cold_fit"][i]), 4),
                        "opportunity_tier": tier, "opportunity_label": label,
                        "cold_fit_tier": cf_tier})
        return out
    # legacy single-head (v0.2/v0.3/v0.4)
    feats = b.get("feature_columns") or list(X.columns)
    p = np.clip(b["model"].predict_proba(X[feats].values)[:, 1], floor, ceil)
    th = b.get("thresholds") or {"risk_low_max": 0.33, "risk_medium_max": 0.66}
    for i in range(len(hd)):
        out.append({"snapshot_date": hd.iloc[i]["SnapshotDate"],
                    "conversion_probability": round(float(p[i]), 4),
                    "risk_tier": _tier(float(p[i]), th)})
    return out


def score_company(query: str, include_history: bool = True) -> dict:
    matches = data.resolve_company(query)
    if not matches:
        return {"contract": "signal-desk-brokerage", "matches": [], "error": f"no company for '{query}'"}
    co = matches[0]
    hmy = int(co["hMy"])
    b = _bundle()
    hist_df = data.feature_history(hmy)
    history = [] if hist_df.empty else _score_history(hist_df, b)
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
    data_as_of = history[-1]["snapshot_date"] if history else None
    fresh = data.feature_freshness()
    out = {"contract": "signal-desk-brokerage", "model_version": b.get("version"),
           "model_source": "standin" if b.get("bundle_format") == "standin" else (b.get("source") or "real-production"),
           "company": {"hMy": hmy, "display_name": co.get("display_name"),
                       "market": co.get("market"), "industry": co.get("zi_primary_industry")},
           "candidates": matches if len(matches) > 1 else [],
           "scorecard": scorecard.scorecard_for_prospect(hmy),  # company-centric headline (via ProspectCompanyLink)
           "current_rules_score": rules,
           "rules_score_stale": rules_stale,
           # Phase 0.2: the rea.* feature pipeline froze in 2026-04; flag it loudly so a
           # stale prediction is never read as current.
           "data_stale": bool(fresh.get("stale")),
           "data_staleness_note": fresh.get("note"),
           "history_count": len(history),
           "data_as_of": data_as_of,
           "history": history if include_history else history[-1:]}
    if b.get("bundle_format") == "brokerage-split-v0.5":
        out["heads"] = {"conversion_probability": "combined (all non-scoop features) — headline estimate",
                        "pipeline_momentum": "CRM stage/activity dynamics — a deal already in motion (strong but not cold signal)",
                        "cold_fit": "firmographics + engagement — the COLD-prospecting signal (honestly weak)",
                        "opportunity_tier": "hot/warm/cold from the combined score distribution (high = pursue)"}
        out["opportunity_tiers"] = b.get("opportunity_tiers")
    return out


def list_companies(limit: int = 50) -> dict:
    return {"contract": "signal-desk-brokerage", "companies": data.list_companies(limit)}
