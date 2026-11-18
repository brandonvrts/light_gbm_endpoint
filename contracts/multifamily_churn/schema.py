"""multifamily-churn contract metadata. STATUS: stub.

Concrete schema lives in app/schemas.py (PredictRequest/Response). This module
holds the contract's static descriptors used by the registry + GET .../schema.
"""
from __future__ import annotations

NAME = "multifamily-churn"
FEATURE_MODE = "identifier-in"
VARIANTS = ["h60", "h180"]

# Yardi data dependency (identifier-in). v6+ serves from data.pit_features_sql (set-based,
# as-of-anchored point-in-time features); the legacy per-resident SP is retained only for
# pre-v5 bundles.
INFERENCE_SP = "data.pit_features_sql"             # as-of point-in-time; legacy: dbo.sp_GetTenantFeaturesByScode
PROPERTY_FILTER = (62, 72, 82, 26, 473)  # CCV / Sierra Pointe / Residences 62W / Winwood / CCV Townhomes

# Bundle invariants the smoke test asserts. v6 adds HasLeaseRecord -> 46 features; earlier
# bundles carry 45 and read their own feature_columns, so the golden test checks the
# DEPLOYED bundle against its own manifest, not this constant.
EXPECTED_FEATURE_COUNT = 46


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
