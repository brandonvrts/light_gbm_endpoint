"""Offline smoke tests (no DB needed). STATUS: stub — mirrors v2/tests/test_smoke.py intent.

Run before any deploy. The bundle test is the key guard against version/arch skew
(critical when serving x86-trained bundles on the ARM64 DGX Spark).
"""
from __future__ import annotations


def test_config_loads_without_secrets_in_source():
    """config.load_db_config raises a clear error when GBME_ keys are absent (no source defaults)."""
    raise NotImplementedError("TODO Phase 2")


def test_multifamily_schema_contract():
    """contracts.multifamily_churn.schema.describe() exposes the expected input/output shape;
    EXPECTED_FEATURE_COUNT == 45."""
    from contracts.multifamily_churn import schema
    assert schema.EXPECTED_FEATURE_COUNT == 45
    assert schema.VARIANTS == ["h60", "h180"]


def test_bundle_loads_and_reproduces_known_prediction():
    """Load the frozen bundle and assert it reproduces a pinned reference prediction within tol.
    Catches library-version / architecture skew. TODO Phase 0/1 once a bundle is frozen."""
    raise NotImplementedError("TODO: load bundles/multifamily_churn/<default>/bundle_h180.joblib")


if __name__ == "__main__":
    test_multifamily_schema_contract()
    print("schema contract OK (other tests are TODO stubs)")
