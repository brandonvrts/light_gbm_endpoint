"""Space-demand contract (PROTOTYPE): ranked in-market relocation/expansion leads."""
from __future__ import annotations
from contracts.signal_desk_spacedemand import data


def space_demand_leads(market: str | None = None, limit: int = 25) -> dict:
    return {"contract": "signal-desk-spacedemand", **data.leads(market, limit)}
