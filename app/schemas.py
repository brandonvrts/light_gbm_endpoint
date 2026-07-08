"""Request bodies for the 3 app contracts."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field


class MultifamilyRequest(BaseModel):
    """Provide either scodes (specific tenants) or property_ids (whole buildings)."""
    scodes: list[str] | None = None
    property_ids: list[int] | None = None
    as_of: str | None = Field(default=None, description="ISO date; default today")
    limit: int | None = Field(default=None, description="optional cap on units scored; default ALL")


class BrokerageRequest(BaseModel):
    company: str = Field(..., description="company hMy (numeric) or name substring")
    include_history: bool = True


class MarketingRequest(BaseModel):
    variant: Literal["cohort", "industry"] = "cohort"
    key: str = Field(..., description="concept key (e.g. healthcare-vertical) or industry name")
