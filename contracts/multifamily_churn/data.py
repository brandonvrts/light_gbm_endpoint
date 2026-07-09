"""Multifamily live data access. identifier-in: SCODEs or buildings -> features via Yardi SP.

Source of truth: InsightMart_Test.dbo.sp_GetTenantFeaturesByScode (returns 64 cols incl.
the 45 model features + ActualChurn label) and Yardi7_Reporting.dbo.TENANT/PROPERTY.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import pandas as pd

from core import db

# The 4 in-scope properties (UNIT.EXCLUDE=0 rentable scope), HPROPERTY -> name.
PROPERTIES = {62: "Country Club Village", 72: "Sierra Pointe",
              82: "Residences at 62W", 26: "Winwood"}

# 45 model features (derived from the SP's raw output; PropertyName/PrimaryPetType encoded).
MODEL_FEATURES = [
    "TenureDaysAtAnchor", "TenureMonths", "DaysToLeaseEnd", "PlannedLeaseDays",
    "IsLeaseExpiringSoon", "LeaseTermMonths", "DecisionNumber", "CurrentRent",
    "ActiveRent", "PriorRent", "RentIncreasePct", "HasNotice", "DaysSinceNotice",
    "NSFCount", "NoPaymentsFlag", "MoveInMonth", "MoveInYear", "DecisionMonth",
    "DecisionYear", "WorkOrderCount", "WorkOrderFrequency", "EmergencyWorkOrders",
    "AvgDaysToComplete", "WorkOrders_Last90Days", "WorkOrders_Last180Days",
    "WorkOrders_Last365Days", "EmergencyWO_Last90Days", "DaysSinceLastWorkOrder",
    "OpenWorkOrdersAtAnchor", "MaxOpenWorkOrderAge", "EarlyWorkOrders30d", "MemoCount",
    "MemoFrequency", "Memos_Last90Days", "Memos_Last180Days", "Memos_Last365Days",
    "DaysSinceLastMemo", "EvictionMemos", "PaymentMemos", "ComplaintMemos",
    "MaintenanceMemos", "PetCount", "HasLargePets", "PropertyName_encoded",
    "PrimaryPetType_encoded",
]
LABEL = "ActualChurn"
ID_COLS = ["SCODE", "FullName", "UnitCode", "PropertyName", "PropertyId",
           "DecisionDate", "MoveOutDate", "LeaseToDate"]


def features_by_scodes(scodes: list[str], as_of: str | None = None, batch: int = 150,
                       workers: int = 4) -> pd.DataFrame:
    """Batched + PARALLEL: FreeTDS truncates a very long @ScodeList param (936 SCODEs
    silently dropped to ~400), so chunk the list; the feature SP is ~0.3s/unit, so run
    the chunks concurrently (each db.query opens its own connection) to cut a full-
    portfolio run ~workers-fold. Order is preserved; each chunk retries on empty."""
    as_of = _safe_asof(as_of)
    clean = [s.strip() for s in scodes if s and s.strip()]
    if not clean:
        return pd.DataFrame()
    chunks = [(i, clean[i:i + batch]) for i in range(0, len(clean), batch)]

    def _fetch(item):
        i, want = item
        lst = ",".join(want)
        for _ in range(3):                       # retry empty/failed chunk (transient under load)
            try:
                d = db.query(
                    f"EXEC [{db.DB_MF}].dbo.sp_GetTenantFeaturesByScode @ScodeList=?, @AsOfDate=?",
                    (lst, as_of))
                if not d.empty:
                    return i, d
            except Exception:
                pass
        raise RuntimeError(f"feature SP returned no rows for scode chunk {i}..{i + batch}")

    with ThreadPoolExecutor(max_workers=min(workers, len(chunks))) as ex:
        parts = list(ex.map(_fetch, chunks))     # raises if any chunk persistently empty
    parts.sort(key=lambda t: t[0])               # restore SCODE order
    return pd.concat([d for _, d in parts], ignore_index=True) if parts else pd.DataFrame()


def _safe_asof(as_of: str | None) -> str:
    """Validate/normalize the as-of date (injection-safe: a parsed date only)."""
    if not as_of:
        return date.today().isoformat()
    try:
        return date.fromisoformat(str(as_of)[:10]).isoformat()
    except Exception:
        return date.today().isoformat()


def active_rentable_scodes(property_ids: list[int], as_of: str | None = None) -> list[str]:
    """The CORRECT cohort: the current resident of each RENTABLE unit at the as-of date.

    Mirrors sp_GetAnchorSnapshot eligibility (active lease at anchor + UNIT.EXCLUDE=0 +
    the 4 properties + valid lease dates), then dedups to ONE current lease per HUNIT
    (latest DTLEASEFROM) so we score occupied rentable units, not every historical/
    overlapping lease record. property_ids are ints and as_of is a parsed date, so
    interpolation here is injection-safe.
    """
    asof = _safe_asof(as_of)
    ids = ",".join(str(int(p)) for p in property_ids) if property_ids else "62,72,82,26"
    q = f"""WITH elig AS (
        SELECT LTRIM(RTRIM(t.SCODE)) AS SCODE,
               ROW_NUMBER() OVER (PARTITION BY t.HUNIT
                                  ORDER BY t.DTLEASEFROM DESC, t.HMYPERSON DESC) AS rn
        FROM [{db.DB_YARDI}].dbo.TENANT t
        WHERE t.DTLEASEFROM IS NOT NULL AND t.DTLEASETO IS NOT NULL AND t.DTMOVEIN IS NOT NULL
          AND t.DTLEASEFROM <= '{asof}' AND t.DTMOVEIN <= '{asof}'
          AND (t.DTMOVEOUT IS NULL OR t.DTMOVEOUT > '{asof}')
          AND t.HPROPERTY IN ({ids})
          AND t.DTLEASETO > '2000-01-01' AND t.DTLEASETO < '2099-01-01'
          AND t.HUNIT IN (SELECT HMY FROM [{db.DB_YARDI}].dbo.[UNIT] WHERE [EXCLUDE] = 0))
    SELECT SCODE FROM elig WHERE rn = 1 AND SCODE IS NOT NULL"""
    df = db.query(q)
    return df["SCODE"].tolist() if not df.empty else []


def scodes_by_lease_window(start: str, end: str, property_ids: list[int] | None = None,
                           as_of: str | None = None) -> list[str]:
    """SCODEs of residents whose CURRENT lease expires (DTLEASETO) within [start, end].

    This is the "lease-expiration window" cohort the renewal-validation workflow used to
    build from a manual Yardi export (e.g. MayLeaseExpirations.xlsx -> Resident Code);
    here it's a direct point-in-time query. Same active-resident eligibility as
    active_rentable_scodes (occupied rentable unit, UNIT.EXCLUDE=0, the 4 properties, one
    current lease per HUNIT) evaluated at `as_of`, PLUS DTLEASETO between start and end.
    All dates are parsed (injection-safe); property_ids are ints.
    """
    s, e = _safe_asof(start), _safe_asof(end)
    if e < s:
        s, e = e, s
    asof = _safe_asof(as_of)
    ids = ",".join(str(int(p)) for p in property_ids) if property_ids else "62,72,82,26"
    q = f"""WITH elig AS (
        SELECT LTRIM(RTRIM(t.SCODE)) AS SCODE,
               ROW_NUMBER() OVER (PARTITION BY t.HUNIT
                                  ORDER BY t.DTLEASEFROM DESC, t.HMYPERSON DESC) AS rn
        FROM [{db.DB_YARDI}].dbo.TENANT t
        WHERE t.DTLEASEFROM IS NOT NULL AND t.DTLEASETO IS NOT NULL AND t.DTMOVEIN IS NOT NULL
          AND t.DTLEASEFROM <= '{asof}' AND t.DTMOVEIN <= '{asof}'
          AND (t.DTMOVEOUT IS NULL OR t.DTMOVEOUT > '{asof}')
          AND t.HPROPERTY IN ({ids})
          AND t.DTLEASETO >= '{s}' AND t.DTLEASETO <= '{e}'
          AND t.HUNIT IN (SELECT HMY FROM [{db.DB_YARDI}].dbo.[UNIT] WHERE [EXCLUDE] = 0))
    SELECT SCODE FROM elig WHERE rn = 1 AND SCODE IS NOT NULL"""
    df = db.query(q)
    return df["SCODE"].tolist() if not df.empty else []


def search_tenants_by_name(q: str, property_ids: list[int] | None = None,
                           as_of: str | None = None, limit: int = 25) -> list[dict]:
    """Resolve a (partial) resident NAME -> current-resident SCODE(s).

    For the chatbot path ("what's the probability John Doe stays") -> map the name to a
    SCODE the model can score. Parameterized LIKE (injection-safe). One current resident
    per rentable unit (dedup per HUNIT), active as of `as_of` (default today).
    """
    asof = _safe_asof(as_of)
    term = (q or "").strip()
    if not term:
        return []
    ids = ",".join(str(int(p)) for p in property_ids) if property_ids else "62,72,82,26"
    like = f"%{term}%"
    sql = f"""WITH elig AS (
        SELECT LTRIM(RTRIM(t.SCODE)) AS SCODE,
               LTRIM(RTRIM(ISNULL(t.SFIRSTNAME,'') + ' ' + ISNULL(t.SLASTNAME,''))) AS FullName,
               t.SUNITCODE AS UnitCode, t.HPROPERTY AS PropertyId,
               ROW_NUMBER() OVER (PARTITION BY t.HUNIT
                                  ORDER BY t.DTLEASEFROM DESC, t.HMYPERSON DESC) AS rn
        FROM [{db.DB_YARDI}].dbo.TENANT t
        WHERE t.DTLEASEFROM IS NOT NULL AND t.DTLEASETO IS NOT NULL AND t.DTMOVEIN IS NOT NULL
          AND t.DTLEASEFROM <= '{asof}' AND t.DTMOVEIN <= '{asof}'
          AND (t.DTMOVEOUT IS NULL OR t.DTMOVEOUT > '{asof}')
          AND t.HPROPERTY IN ({ids})
          AND (ISNULL(t.SFIRSTNAME,'') + ' ' + ISNULL(t.SLASTNAME,'')) LIKE ?
          AND t.HUNIT IN (SELECT HMY FROM [{db.DB_YARDI}].dbo.[UNIT] WHERE [EXCLUDE] = 0))
    SELECT TOP ({int(limit)}) SCODE, FullName, UnitCode, PropertyId
    FROM elig WHERE rn = 1 AND SCODE IS NOT NULL ORDER BY FullName"""
    df = db.query(sql, (like,))
    if df.empty:
        return []
    out = []
    for _, r in df.iterrows():
        pid = r.get("PropertyId")
        out.append({"scode": str(r["SCODE"]).strip(),
                    "full_name": r.get("FullName"),
                    "unit": r.get("UnitCode"),
                    "property": PROPERTIES.get(int(pid)) if pid is not None else None})
    return out


def list_properties() -> list[dict]:
    return [{"property_id": k, "name": v} for k, v in PROPERTIES.items()]
