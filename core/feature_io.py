"""The data seam: identifier-in (Yardi SP) vs feature-in (supplied vector). STATUS: stub.

This is the ONLY place the Yardi read-only DB credential is used (config.load_db_config).
Isolating it here keeps data-access coupling contained (Risk 4 in the design docs).
"""
from __future__ import annotations
from typing import Any

# import pyodbc, pandas as pd
# from config import load_db_config


def fetch_identifier_in(sp: str, scodes: list[str], as_of_date: str) -> "Any":
    """Run a contract's inference stored procedure for IDs + as-of date, return raw rows.

    For multifamily-churn: sp = 'dbo.sp_GetTenantFeaturesByScode',
        EXEC sp @ScodeList=?, @AsOfDate=?  with ','.join(scodes).
    TODO Phase 2:
        cfg = load_db_config(); cn = pyodbc.connect(cfg.connection_string())
        cur.execute(f"EXEC {sp} @ScodeList=?, @AsOfDate=?", ",".join(scodes), as_of_date)
        return pd.DataFrame.from_records(cur.fetchall(), columns=[c[0] for c in cur.description])
    """
    raise NotImplementedError("TODO Phase 2: pyodbc + EXEC <sp>")


def validate_feature_in(features: list[dict], input_schema: dict) -> "Any":
    """feature-in path (marketing, optionally signal-desk): validate a supplied vector
    against the contract's schema and hand it straight to the model. No DB call.
    TODO when first feature-in contract is onboarded.
    """
    raise NotImplementedError
