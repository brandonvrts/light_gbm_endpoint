"""Request bodies for the 3 app contracts."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field


class MultifamilyRequest(BaseModel):
    """Provide ONE of: scode (single tenant), scodes (specific tenants),
    expires_from+expires_to (lease-expiration window), or property_ids (whole buildings)."""
    scode: str | None = Field(default=None, description="single tenant SCODE, e.g. t0014834")
    scodes: list[str] | None = None
    expires_from: str | None = Field(default=None, description="ISO date; with expires_to, scores residents whose lease expires in [from, to]")
    expires_to: str | None = None
    property_ids: list[int] | None = None
    as_of: str | None = Field(default=None, description="ISO date; default today")
    limit: int | None = Field(default=None, description="optional cap on units scored; default ALL")


class BrokerageRequest(BaseModel):
    company: str = Field(..., description="company hMy (numeric) or name substring")
    include_history: bool = True


class MarketingRequest(BaseModel):
    variant: Literal["cohort", "industry"] = "cohort"
    key: str = Field(..., description="concept key (e.g. healthcare-vertical) or industry name")
