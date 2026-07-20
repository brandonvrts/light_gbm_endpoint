"""Marketing aggregation: selectable cohort (concept vertical) or industry ->
aggregate model outputs (counts, decile distribution, by-market, cohort win-rates).
Plus cohort views over the space-demand ML bundle (space_demand module)."""
from __future__ import annotations
from contracts.marketing import data, space_demand as sdm


def options() -> dict:
    return {"contract": "marketing", **data.list_options(),
            "space_demand": sdm.cohort_options()}


def space_demand(market: str | None = None, min_scored: int = 3) -> dict:
    """Per-cohort 'who needs space' rollup (same frozen spacedemand bundle as brokerage)."""
    return {"contract": "marketing", **sdm.cohort_rollup(market, min_scored)}


def space_demand_leads(cohort: str, market: str | None = None, limit: int = 25) -> dict:
    """Ranked space-need leads inside one marketing cohort."""
    return {"contract": "marketing", **sdm.cohort_leads(cohort, market, limit)}


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
