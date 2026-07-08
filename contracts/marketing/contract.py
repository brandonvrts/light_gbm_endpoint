"""Marketing aggregation: selectable cohort (concept vertical) or industry ->
aggregate model outputs (counts, decile distribution, by-market, cohort win-rates)."""
from __future__ import annotations
from contracts.marketing import data


def options() -> dict:
    return {"contract": "marketing", **data.list_options()}


def targeting() -> dict:
    """Which cohorts/verticals to reach out to, ranked by expected opportunity."""
    return {"contract": "marketing", **data.cohort_targeting()}


def aggregate(variant: str, key: str) -> dict:
    if variant == "industry":
        agg = data.industry_aggregate(key)
    else:
        agg = data.cohort_aggregate(key)
    # attach the deal win-rate table for the matching vertical (best-effort)
    wr = data.cohort_win_rates()
    agg["cohort_win_rates"] = [] if wr.empty else wr.to_dict("records")
    agg["company_count"] = int(sum((m.get("n") or 0) for m in agg.get("by_market", [])))
    agg["contract"] = "marketing"
    agg["variant"] = variant
    agg["model_source"] = "standin"
    return agg
