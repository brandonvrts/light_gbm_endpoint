"""multifamily-churn contract metadata. STATUS: stub.

Concrete schema lives in app/schemas.py (PredictRequest/Response). This module
holds the contract's static descriptors used by the registry + GET .../schema.
"""
from __future__ import annotations

NAME = "multifamily-churn"
FEATURE_MODE = "identifier-in"
VARIANTS = ["h60", "h180"]

# Yardi data dependency (identifier-in)
INFERENCE_SP = "dbo.sp_GetTenantFeaturesByScode"   # @ScodeList, @AsOfDate, @WindowMonths=18
PROPERTY_FILTER = (62, 72, 82, 26)  # CCV / Sierra Pointe / Residences 62W / Winwood

# Bundle invariants the smoke test asserts against the frozen artifact:
EXPECTED_FEATURE_COUNT = 45
# Pin these from the actually-frozen bundle's manifest (see docs/ARCHITECTURE.md §4):
#   h180 deployment bundle = LightGBM + isotonic + 32.3% anchor, clip [0.01, 0.99]
#   h60  = ranker (use for ranking only)


def describe() -> dict:
    """Input/output schema summary for GET /contracts/multifamily-churn/schema. TODO Phase 1."""
    return {
        "name": NAME,
        "feature_mode": FEATURE_MODE,
        "variants": VARIANTS,
        "input": {"scode?": "str (single tenant)", "scodes?": "list[str]",
                  "expires_from?/expires_to?": "date (lease-expiration window)",
                  "property_ids?": "list[int]", "as_of?": "date", "limit?": "int"},
        "output": {"model_version": "str", "as_of": "date",
                   "results": [{"scode": "str", "churn_probability": "float", "risk_tier": "low|medium|high"}]},
    }
