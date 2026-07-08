"""Brokerage (Signal Desk) live data access. Pulls a company's full historical
feature snapshots from InsightMart.rea so the model can run through all of them.
"""
from __future__ import annotations
import pandas as pd
from core import db
from contracts.signal_desk_brokerage.schema import FEATURES

_FCOLS = ", ".join(f"f.[{c}]" for c in FEATURES)


def resolve_company(q: str) -> list[dict]:
    """Resolve a company by hMy (numeric) or name -> [{hMy, display_name, market, industry}]."""
    q = (q or "").strip()
    if q.isdigit():
        df = db.query(
            f"SELECT TOP 5 hMy, display_name, market, zi_primary_industry "
            f"FROM [{db.DB_MAIN}].rea.vw_ProspectEnriched WHERE hMy=?", (int(q),))
    else:
        df = db.query(
            f"SELECT TOP 15 hMy, display_name, market, zi_primary_industry "
            f"FROM [{db.DB_MAIN}].rea.vw_ProspectEnriched "
            f"WHERE display_name LIKE ? OR zi_company_name LIKE ? ORDER BY display_name",
            (f"%{q}%", f"%{q}%"))
    return [] if df.empty else df.to_dict("records")


def list_companies(limit: int = 2000) -> list[dict]:
    """ALL prospects that have any feature-snapshot history, alphabetical so the full
    population is selectable (the search box resolves any company by id/name)."""
    df = db.query(
        f"SELECT TOP {int(limit)} e.hMy, e.display_name, e.market, e.zi_primary_industry, "
        f"COUNT(f.SnapshotDate) AS snapshots "
        f"FROM [{db.DB_MAIN}].rea.vw_ProspectEnriched e "
        f"JOIN [{db.DB_MAIN}].rea.ProspectFeatureSnapshot f ON f.hMy=e.hMy "
        f"GROUP BY e.hMy, e.display_name, e.market, e.zi_primary_industry "
        f"HAVING COUNT(f.SnapshotDate) >= 1 ORDER BY e.display_name")
    return [] if df.empty else df.to_dict("records")


def feature_history(hmy: int) -> pd.DataFrame:
    """All snapshots for a company, oldest->newest (the historical-prediction series)."""
    return db.query(
        f"SELECT f.hMy, f.SnapshotDate, {_FCOLS} "
        f"FROM [{db.DB_MAIN}].rea.ProspectFeatureSnapshot f "
        f"WHERE f.hMy=? ORDER BY f.SnapshotDate", (int(hmy),))


def current_rules_score(hmy: int) -> dict | None:
    df = db.query(
        f"SELECT TOP 1 Probability, Decile, ScoredAt FROM [{db.DB_MAIN}].rea.ProspectScore "
        f"WHERE hMy=? ORDER BY ScoredAt DESC", (int(hmy),))
    return None if df.empty else df.iloc[0].to_dict()


def training_data(limit: int = 8000) -> pd.DataFrame:
    """Real labeled rows: ProspectFeatureSnapshot joined to ProspectLabelSnapshot."""
    return db.query(
        f"SELECT TOP {int(limit)} {_FCOLS}, l.Label "
        f"FROM [{db.DB_MAIN}].rea.ProspectFeatureSnapshot f "
        f"JOIN [{db.DB_MAIN}].rea.ProspectLabelSnapshot l "
        f"  ON l.hMy=f.hMy AND l.SnapshotDate=f.SnapshotDate "
        f"WHERE l.Label IS NOT NULL")
