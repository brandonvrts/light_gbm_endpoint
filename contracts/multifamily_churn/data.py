"""Multifamily live data access. identifier-in: SCODEs or buildings -> features via Yardi SP.

Source of truth: InsightMart_Test.dbo.sp_GetTenantFeaturesByScode (returns 64 cols incl.
the 45 model features + ActualChurn label) and Yardi7_Reporting.dbo.TENANT/PROPERTY.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import numpy as np
import pandas as pd

from core import db

# In-scope properties (UNIT.EXCLUDE=0 rentable scope), HPROPERTY -> name.
# 473 (CCV Townhomes) is part of the portfolio's renewal reporting but was NOT in the
# model's training scope -- its residents score with the property-encoding fallback.
PROPERTIES = {62: "Country Club Village", 72: "Sierra Pointe",
              82: "Residences at 62W", 26: "Winwood",
              473: "Country Club Village Townhomes"}
DEFAULT_PROPERTY_IDS = list(PROPERTIES)

# 46 model features. HasLeaseRecord (v6+) flags residents with NO Lease_History row at
# the anchor (~25% of actives) so they can be trained/scored honestly instead of silently
# median-filled; earlier bundles carry 45 features and read their own feature_columns, so
# both coexist. PropertyName/PrimaryPetType are label-encoded.
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
    "MaintenanceMemos", "PetCount", "HasLargePets", "HasLeaseRecord",
    "PropertyName_encoded", "PrimaryPetType_encoded",
]
LABEL = "ActualChurn"
ID_COLS = ["SCODE", "FullName", "UnitCode", "PropertyName", "PropertyId",
           "DecisionDate", "MoveOutDate", "LeaseToDate"]

# TENANT.ISTATUS values that denote a tenancy that occupies (or will occupy) a unit:
# 0 current, 1 past, 2 future, 3 eviction, 4 notice. 6/7/9 are canceled/denied
# APPLICATIONS: they carry a planned DTMOVEIN and (almost) never a DTMOVEOUT, so every
# DTMOVEIN/DTMOVEOUT eligibility filter admits them unless ISTATUS is checked. The
# 2026-09-25 audit found them to be ~50% of any anchor population (959 of 1,901 at
# 2024-06-30), 100% of the HasLeaseRecord=0 rows, 35 of 950 served scodes (8 of which
# displaced a real resident in the per-unit dedup) and the bogus "next move-in" that
# truncated ledger vacancy gaps. ISTATUS is the CURRENT status, so past residents are 1
# today -- keep 1 in the set or historical as-of populations lose everyone who left.
OCCUPANCY_STATUSES = (0, 1, 2, 3, 4)
OCC_SQL = "ISTATUS IN (0,1,2,3,4)"

# Curated memo-category keyword sets (v6, Phase 2.4). The old serve/train SP matched bare
# %notice% (counting benign "notice to enter"/"courtesy notice" as EVICTION) and %emergency%
# (matching emergency-contact memos as MAINTENANCE). These phrases are precise; a memo may
# match more than one category (independent SUMs, matching prior behavior). Constants only
# -- built into SQL as literals (injection-safe).
MEMO_CATEGORIES: dict[str, list[str]] = {
    "EvictionMemos": ["evict", "notice to vacate", "notice to quit", "lease violation",
                      "terminate lease", "non-renew", "nonrenew", "pay or quit", "3 day",
                      "3-day"],
    "PaymentMemos": ["late fee", "past due", "delinquent", "nsf", "returned payment",
                     "balance due", "collection", "insufficient funds", "bounced"],
    "ComplaintMemos": ["complaint", "noise", "disturbance", "police", "pet violation",
                       "parking violation", "lease violation", "unauthorized"],
    "MaintenanceMemos": ["repair", "maintenance request", "broken", "leak", "hvac",
                         "appliance", "not working", "clogged", "no hot water"],
}


def _memo_case_sql(alias: str = "m") -> str:
    """SUM(CASE ...) fragments for the memo categories, from MEMO_CATEGORIES."""
    parts = []
    for cat, kws in MEMO_CATEGORIES.items():
        conds = " OR ".join(f"{alias}.STEXT LIKE '%{kw}%'" for kw in kws)
        parts.append(f"SUM(CASE WHEN {conds} THEN 1 ELSE 0 END) AS {cat}")
    return ",\n            ".join(parts)


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


def pit_features_sql(asof: str, scodes: list[str] | None = None,
                     property_ids: list[int] | None = None, horizon: int | None = None,
                     window_months: int = 18) -> str:
    """Point-in-time feature snapshot at `asof` -- the SINGLE source of truth for both
    training-anchor extraction and cohort serving (Phase 2.1: train/serve parity by
    construction). Read-only.

    Eligibility uses only STABLE columns (DTMOVEIN/DTMOVEOUT, set once) -- never the
    Yardi-mutated DTLEASEFROM/DTLEASETO -- so historical anchors keep residents who later
    renewed. All lease timing/rent is read from Lease_History as of `asof`.

      horizon      -- if given, adds the MoveOutInHorizon training label (churn within
                      `horizon` days after asof); omit for serving.
      scodes       -- restrict to a cohort (serving); omit for the full active population.
      LEFT JOIN lease history + HasLeaseRecord (Phase 2.2): residents with no lease row at
                      the anchor (~25%) are KEPT with HasLeaseRecord=0 instead of dropped.
      memo categories come from MEMO_CATEGORIES (Phase 2.4).

    `asof` is validated ISO; ids are ints; scodes are single-quote-stripped (injection-safe)."""
    a = _safe_asof(asof)
    w = int(window_months)
    ws = (date.fromisoformat(a) - timedelta(days=int(round(w * 30.44)))).isoformat()
    pids = property_ids if property_ids else DEFAULT_PROPERTY_IDS
    idlist = ",".join(str(int(p)) for p in pids)
    Y = f"[{db.DB_YARDI}].dbo"
    scode_filter = ""
    if scodes is not None:
        safe = [s.strip().replace("'", "") for s in scodes if s and s.strip()]
        inlist = ",".join(f"'{s}'" for s in safe) or "''"
        scode_filter = f"AND LTRIM(RTRIM(t.SCODE)) IN ({inlist})"
    if horizon is not None:
        he = (date.fromisoformat(a) + timedelta(days=int(horizon))).isoformat()
        label_col = (f"CASE WHEN bt.DTMOVEOUT IS NOT NULL AND bt.DTMOVEOUT > '{a}' "
                     f"AND bt.DTMOVEOUT <= '{he}' THEN 1 ELSE 0 END AS MoveOutInHorizon,")
    else:
        label_col = "CAST(NULL AS INT) AS MoveOutInHorizon,"
    memo_cases = _memo_case_sql("m")
    return f"""
    WITH base_tenants AS (
        SELECT t.HMYPERSON, t.SCODE, t.SUNITCODE, t.HPROPERTY, t.HUNIT,
               t.SFIRSTNAME, t.SLASTNAME,
               t.DTMOVEIN, t.DTMOVEOUT, t.DTNOTICE, t.iNSFCount, t.BNOPAYMENTS
        FROM {Y}.TENANT t
        WHERE t.DTMOVEIN IS NOT NULL AND t.DTMOVEIN <= '{a}'
          AND (t.DTMOVEOUT IS NULL OR t.DTMOVEOUT > '{a}')
          AND t.HPROPERTY IN ({idlist})
          AND t.{OCC_SQL}
          AND t.HUNIT IN (SELECT HMY FROM {Y}.UNIT WHERE EXCLUDE = 0)
          {scode_filter}
    ),
    lease_pit AS (
        SELECT bt.HMYPERSON, lh.dtLeaseFrom, lh.dtLeaseTo, lh.iTerm,
               ROW_NUMBER() OVER (PARTITION BY bt.HMYPERSON
                                  ORDER BY lh.dtLeaseFrom DESC, lh.dtOccurred DESC,
                                           lh.hMy DESC) AS rn
        FROM base_tenants bt
        JOIN {Y}.Lease_History lh ON lh.hTent = bt.HMYPERSON
        WHERE lh.sStatus IN ('Past', 'Current') AND lh.dtLeaseFrom <= '{a}'
    ),
    lease_pit_final AS (SELECT * FROM lease_pit WHERE rn = 1),
    rent_pit AS (
        SELECT bt.HMYPERSON, lh.cRent AS ActiveRent, lh.dtLeaseFrom AS ActiveLeaseFrom,
               ROW_NUMBER() OVER (PARTITION BY bt.HMYPERSON
                                  ORDER BY lh.dtLeaseFrom DESC) AS rn
        FROM base_tenants bt
        JOIN {Y}.Lease_History lh ON lh.hTent = bt.HMYPERSON
        WHERE lh.cRent > 0 AND lh.dtLeaseFrom <= '{a}' AND lh.sStatus IN ('Past', 'Current')
    ),
    rent_pit_final AS (SELECT * FROM rent_pit WHERE rn = 1),
    prior_rent AS (
        SELECT rp.HMYPERSON, lh.cRent AS PriorRent,
               ROW_NUMBER() OVER (PARTITION BY rp.HMYPERSON
                                  ORDER BY lh.dtLeaseFrom DESC) AS rn
        FROM rent_pit_final rp
        JOIN {Y}.Lease_History lh ON lh.hTent = rp.HMYPERSON
        WHERE lh.cRent > 0 AND lh.dtLeaseFrom < rp.ActiveLeaseFrom
          AND lh.sStatus IN ('Past', 'Current')
    ),
    prior_rent_final AS (SELECT HMYPERSON, PriorRent FROM prior_rent WHERE rn = 1),
    wo_agg AS (
        SELECT bt.HMYPERSON,
            COUNT(*) AS WorkOrderCount,
            SUM(CASE WHEN wo.spriority LIKE '%emergency%' THEN 1 ELSE 0 END) AS EmergencyWorkOrders,
            AVG(CAST(CASE WHEN wo.dtwcompl IS NOT NULL AND wo.dtwcompl <= '{a}'
                          THEN DATEDIFF(DAY, wo.dtcall, wo.dtwcompl) END AS FLOAT)) AS AvgDaysToComplete,
            SUM(CASE WHEN wo.dtcall >= DATEADD(DAY, -90,  '{a}') THEN 1 ELSE 0 END) AS WorkOrders_Last90Days,
            SUM(CASE WHEN wo.dtcall >= DATEADD(DAY, -180, '{a}') THEN 1 ELSE 0 END) AS WorkOrders_Last180Days,
            SUM(CASE WHEN wo.dtcall >= DATEADD(DAY, -365, '{a}') THEN 1 ELSE 0 END) AS WorkOrders_Last365Days,
            SUM(CASE WHEN wo.dtcall >= DATEADD(DAY, -90,  '{a}')
                      AND wo.spriority LIKE '%emergency%' THEN 1 ELSE 0 END) AS EmergencyWO_Last90Days,
            DATEDIFF(DAY, MAX(wo.dtcall), '{a}') AS DaysSinceLastWorkOrder,
            SUM(CASE WHEN (wo.dtwcompl IS NULL OR wo.dtwcompl > '{a}')
                      AND wo.dtcall <= '{a}' THEN 1 ELSE 0 END) AS OpenWorkOrdersAtAnchor,
            MAX(CASE WHEN (wo.dtwcompl IS NULL OR wo.dtwcompl > '{a}') AND wo.dtcall <= '{a}'
                     THEN DATEDIFF(DAY, wo.dtcall, '{a}') ELSE 0 END) AS MaxOpenWorkOrderAge,
            SUM(CASE WHEN wo.dtcall <= DATEADD(DAY, 30, bt.DTMOVEIN)
                      AND wo.dtcall <= '{a}' THEN 1 ELSE 0 END) AS EarlyWorkOrders30d
        FROM base_tenants bt
        JOIN {Y}.mm2wo wo ON wo.htenant = bt.HMYPERSON
        WHERE wo.dtcall BETWEEN '{ws}' AND '{a}'
        GROUP BY bt.HMYPERSON
    ),
    memo_agg AS (
        SELECT bt.HMYPERSON,
            COUNT(*) AS MemoCount,
            SUM(CASE WHEN m.UDATE >= DATEADD(DAY, -90,  '{a}') THEN 1 ELSE 0 END) AS Memos_Last90Days,
            SUM(CASE WHEN m.UDATE >= DATEADD(DAY, -180, '{a}') THEN 1 ELSE 0 END) AS Memos_Last180Days,
            SUM(CASE WHEN m.UDATE >= DATEADD(DAY, -365, '{a}') THEN 1 ELSE 0 END) AS Memos_Last365Days,
            DATEDIFF(DAY, MAX(m.UDATE), '{a}') AS DaysSinceLastMemo,
            {memo_cases}
        FROM base_tenants bt
        JOIN {Y}.MEMO m ON m.HFILERCD = bt.HMYPERSON
        WHERE m.SSTATUS = 'Memo' AND m.IFILETYPE = '1'
          AND m.UDATE BETWEEN '{ws}' AND '{a}' AND m.STEXT IS NOT NULL
        GROUP BY bt.HMYPERSON
    ),
    pet_agg AS (
        SELECT hMyPerson, PetCount, HasLargePets, PrimaryPetType
        FROM (SELECT pp.hMyPerson,
                     COUNT(*) OVER (PARTITION BY pp.hMyPerson) AS PetCount,
                     MAX(CASE WHEN pp.dPetWeight > 50 THEN 1 ELSE 0 END)
                         OVER (PARTITION BY pp.hMyPerson) AS HasLargePets,
                     pp.sPetType AS PrimaryPetType,
                     ROW_NUMBER() OVER (PARTITION BY pp.hMyPerson ORDER BY pp.sPetType) AS rn
              FROM {Y}.Person_Pet pp
              WHERE pp.hMyPerson IS NOT NULL AND pp.hMyPerson != 0) ranked
        WHERE rn = 1
    )
    SELECT
        bt.HMYPERSON AS hMyPerson, LTRIM(RTRIM(bt.SCODE)) AS SCODE, '{a}' AS AnchorDate,
        LTRIM(RTRIM(ISNULL(bt.SFIRSTNAME,'') + ' ' + ISNULL(bt.SLASTNAME,''))) AS FullName,
        bt.SUNITCODE AS UnitCode, bt.HPROPERTY AS PropertyId,
        bt.DTMOVEIN AS MoveInDate, bt.DTMOVEOUT AS MoveOutDate,
        lp.dtLeaseFrom AS LeaseFromDate, lp.dtLeaseTo AS LeaseToDate,
        CASE WHEN lp.HMYPERSON IS NULL THEN 0 ELSE 1 END AS HasLeaseRecord,
        ISNULL(rp.ActiveRent, 0) AS CurrentRent, rp.ActiveRent AS ActiveRent,
        pr.PriorRent AS PriorRent,
        CASE WHEN pr.PriorRent IS NOT NULL AND pr.PriorRent > 0
             THEN (rp.ActiveRent - pr.PriorRent) / pr.PriorRent * 100.0 END AS RentIncreasePct,
        lp.iTerm AS LeaseTermMonths,
        CASE WHEN bt.DTNOTICE IS NOT NULL AND bt.DTNOTICE <= '{a}'
             THEN DATEDIFF(DAY, bt.DTNOTICE, '{a}') END AS DaysSinceNotice,
        CASE WHEN bt.DTNOTICE IS NOT NULL AND bt.DTNOTICE <= '{a}' THEN 1 ELSE 0 END AS HasNotice,
        ISNULL(CAST(bt.iNSFCount AS INT), 0) AS NSFCount,
        CASE WHEN bt.BNOPAYMENTS = 1 THEN 1 ELSE 0 END AS NoPaymentsFlag,
        {label_col}
        DATEPART(MONTH, '{a}') AS DecisionMonth, DATEPART(YEAR, '{a}') AS DecisionYear,
        ISNULL(wo.WorkOrderCount, 0) AS WorkOrderCount,
        ISNULL(wo.EmergencyWorkOrders, 0) AS EmergencyWorkOrders,
        ISNULL(wo.AvgDaysToComplete, 0) AS AvgDaysToComplete,
        ISNULL(wo.WorkOrders_Last90Days, 0) AS WorkOrders_Last90Days,
        ISNULL(wo.WorkOrders_Last180Days, 0) AS WorkOrders_Last180Days,
        ISNULL(wo.WorkOrders_Last365Days, 0) AS WorkOrders_Last365Days,
        ISNULL(wo.EmergencyWO_Last90Days, 0) AS EmergencyWO_Last90Days,
        ISNULL(wo.DaysSinceLastWorkOrder, 9999) AS DaysSinceLastWorkOrder,
        ISNULL(wo.OpenWorkOrdersAtAnchor, 0) AS OpenWorkOrdersAtAnchor,
        ISNULL(wo.MaxOpenWorkOrderAge, 0) AS MaxOpenWorkOrderAge,
        ISNULL(wo.EarlyWorkOrders30d, 0) AS EarlyWorkOrders30d,
        ISNULL(memo.MemoCount, 0) AS MemoCount,
        ISNULL(memo.Memos_Last90Days, 0) AS Memos_Last90Days,
        ISNULL(memo.Memos_Last180Days, 0) AS Memos_Last180Days,
        ISNULL(memo.Memos_Last365Days, 0) AS Memos_Last365Days,
        ISNULL(memo.DaysSinceLastMemo, 9999) AS DaysSinceLastMemo,
        ISNULL(memo.EvictionMemos, 0) AS EvictionMemos,
        ISNULL(memo.PaymentMemos, 0) AS PaymentMemos,
        ISNULL(memo.ComplaintMemos, 0) AS ComplaintMemos,
        ISNULL(memo.MaintenanceMemos, 0) AS MaintenanceMemos,
        CAST(ISNULL(wo.WorkOrderCount, 0) AS FLOAT) / {w} * 12 AS WorkOrderFrequency,
        CAST(ISNULL(memo.MemoCount, 0) AS FLOAT) / {w} * 12 AS MemoFrequency,
        ISNULL(pet.PetCount, 0) AS PetCount,
        ISNULL(pet.HasLargePets, 0) AS HasLargePets,
        ISNULL(pet.PrimaryPetType, 'None') AS PrimaryPetType
    FROM base_tenants bt
    LEFT JOIN lease_pit_final lp ON lp.HMYPERSON = bt.HMYPERSON
    LEFT JOIN rent_pit_final rp ON rp.HMYPERSON = bt.HMYPERSON
    LEFT JOIN prior_rent_final pr ON pr.HMYPERSON = bt.HMYPERSON
    LEFT JOIN wo_agg wo ON wo.HMYPERSON = bt.HMYPERSON
    LEFT JOIN memo_agg memo ON memo.HMYPERSON = bt.HMYPERSON
    LEFT JOIN pet_agg pet ON pet.hMyPerson = bt.HMYPERSON"""


def cohort_features_asof(scodes: list[str], as_of: str | None = None,
                         property_ids: list[int] | None = None,
                         batch: int = 400) -> pd.DataFrame:
    """Set-based point-in-time feature extraction for a cohort of SCODEs (Phase 2.1).

    Replaces the per-resident sp_GetTenantFeaturesByScode for v5+/v6 serving: ONE query
    per (chunked) cohort instead of ~0.3s/resident, and -- crucially -- features are
    anchored at `as_of`, not COALESCE(DTMOVEOUT, DTLEASETO), so scoring a PAST month no
    longer leaks the outcome into a decided resident's features. Same SQL as training."""
    asof = _safe_asof(as_of)
    clean = [s.strip() for s in scodes if s and s.strip()]
    if not clean:
        return pd.DataFrame()
    parts = []
    for i in range(0, len(clean), batch):
        chunk = clean[i:i + batch]
        d = db.query(pit_features_sql(asof, scodes=chunk, property_ids=property_ids))
        if not d.empty:
            parts.append(d)
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    # preserve requested order
    order = {s.lower(): k for k, s in enumerate(clean)}
    df["_o"] = df["SCODE"].str.strip().str.lower().map(order)
    return df.sort_values("_o").drop(columns="_o").reset_index(drop=True)


def active_rentable_scodes(property_ids: list[int], as_of: str | None = None) -> list[str]:
    """The CORRECT cohort: the current resident of each RENTABLE unit at the as-of date.

    Mirrors sp_GetAnchorSnapshot eligibility (active lease at anchor + UNIT.EXCLUDE=0 +
    the 4 properties + valid lease dates), then dedups to ONE current lease per HUNIT
    (latest DTLEASEFROM) so we score occupied rentable units, not every historical/
    overlapping lease record. property_ids are ints and as_of is a parsed date, so
    interpolation here is injection-safe.
    """
    asof = _safe_asof(as_of)
    ids = ",".join(str(int(p)) for p in property_ids) if property_ids else ",".join(str(p) for p in DEFAULT_PROPERTY_IDS)
    q = f"""WITH elig AS (
        SELECT LTRIM(RTRIM(t.SCODE)) AS SCODE,
               ROW_NUMBER() OVER (PARTITION BY t.HUNIT
                                  ORDER BY t.DTLEASEFROM DESC, t.HMYPERSON DESC) AS rn
        FROM [{db.DB_YARDI}].dbo.TENANT t
        WHERE t.DTLEASEFROM IS NOT NULL AND t.DTLEASETO IS NOT NULL AND t.DTMOVEIN IS NOT NULL
          AND t.DTLEASEFROM <= '{asof}' AND t.DTMOVEIN <= '{asof}'
          AND (t.DTMOVEOUT IS NULL OR t.DTMOVEOUT > '{asof}')
          AND t.HPROPERTY IN ({ids})
          AND t.{OCC_SQL}
          AND t.DTLEASETO > '2000-01-01' AND t.DTLEASETO < '2099-01-01'
          AND t.HUNIT IN (SELECT HMY FROM [{db.DB_YARDI}].dbo.[UNIT] WHERE [EXCLUDE] = 0))
    SELECT SCODE FROM elig WHERE rn = 1 AND SCODE IS NOT NULL"""
    df = db.query(q)
    return df["SCODE"].tolist() if not df.empty else []


def lease_window_cohort(start: str, end: str, property_ids: list[int] | None = None,
                        as_of: str | None = None) -> pd.DataFrame:
    """Residents whose lease -- the one ACTIVE at `as_of` -- expires within [start, end].

    Source of truth is Lease_History, NOT TENANT.DTLEASETO: Yardi updates DTLEASETO in
    place when a tenant renews, so a past month's expirations silently shrink to only the
    not-yet-renewed residents (April 2026: 5 of 79). Lease_History keeps every lease term
    as an immutable row; the active lease at `as_of` = the latest real row (sStatus
    Past/Current -- Denied/Approved rows are unaccepted renewal PROPOSALS) with
    dtLeaseFrom <= as_of. Matches the manual Yardi "<Month>LeaseExpirations" export the
    renewal-validation workflow used (73/79 April 2026; misses are edge cases with no
    Lease_History row for the expiring term).

    Move-outs stay in the cohort (they ARE the churn outcome), including early
    terminations up to 90 days before the window opens (e.g. left in March against an
    April lease-end). Older move-outs are mid-lease breaks that belong to the month they
    happened, not to this expiration month. Returns SCODE, HMYPERSON, LeaseTo, MoveOut,
    PropertyId. All dates are parsed and ids are ints (injection-safe).
    """
    s, e = _safe_asof(start), _safe_asof(end)
    if e < s:
        s, e = e, s
    grace = (date.fromisoformat(s) - timedelta(days=90)).isoformat()
    # Default the eligibility date to min(today, window start): a PAST window is scored
    # point-in-time at its start (before renewals mutated the active lease); a FUTURE
    # window uses today's knowledge. Defaulting to today would collapse a past cohort to
    # only the residents who never renewed (April 2026: 26 of 93).
    asof = _safe_asof(as_of) if as_of else min(date.today().isoformat(), s)
    pids = property_ids if property_ids else DEFAULT_PROPERTY_IDS
    ids = ",".join(str(int(p)) for p in pids)
    q = f"""WITH active_lease AS (
        SELECT lh.hTent, lh.dtLeaseTo,
               ROW_NUMBER() OVER (PARTITION BY lh.hTent
                                  ORDER BY lh.dtLeaseFrom DESC, lh.dtOccurred DESC,
                                           lh.hMy DESC) AS rn
        FROM [{db.DB_YARDI}].dbo.Lease_History lh
        WHERE lh.sStatus IN ('Past', 'Current') AND lh.dtLeaseFrom <= '{asof}')
    SELECT DISTINCT LTRIM(RTRIM(t.SCODE)) AS SCODE, t.HMYPERSON,
           al.dtLeaseTo AS LeaseTo, t.DTMOVEOUT AS MoveOut, t.HPROPERTY AS PropertyId
    FROM [{db.DB_YARDI}].dbo.TENANT t
    JOIN active_lease al ON al.hTent = t.HMYPERSON AND al.rn = 1
    WHERE t.HPROPERTY IN ({ids}) AND t.{OCC_SQL}
      AND t.DTMOVEIN IS NOT NULL AND t.DTMOVEIN <= '{e}'
      AND (t.DTMOVEOUT IS NULL OR t.DTMOVEOUT >= '{grace}')
      AND al.dtLeaseTo >= '{s}' AND al.dtLeaseTo <= '{e}'
      AND t.HUNIT IN (SELECT HMY FROM [{db.DB_YARDI}].dbo.[UNIT] WHERE [EXCLUDE] = 0)"""
    df = db.query(q)
    return df[df["SCODE"].notna()] if not df.empty else df


def scodes_by_lease_window(start: str, end: str, property_ids: list[int] | None = None,
                           as_of: str | None = None) -> list[str]:
    """SCODEs of residents whose active-at-`as_of` lease expires within [start, end]."""
    df = lease_window_cohort(start, end, property_ids, as_of)
    return df["SCODE"].tolist() if not df.empty else []


def renewal_outcomes(hmypersons: list[int], window_start: str,
                     window_end: str) -> dict[int, str]:
    """Classify each cohort member's renewal outcome as of TODAY:
      transferred -- DTMOVEOUT is set but it's an internal unit transfer (person stayed at
                     the property) -- NOT a churn (Phase 1.1)
      churned  -- DTMOVEOUT is set and it's a real departure
      renewed  -- DTLEASETO now extends past the window, OR a signed 'Scheduled'/'Current'
                  Lease_History row starts after the window opened
      pending  -- none of the above (offer outstanding, or holdover/MTM)
    """
    if not hmypersons:
        return {}
    s, e = _safe_asof(window_start), _safe_asof(window_end)
    transfers = transfer_origin_hmypersons()
    ids = ",".join(str(int(h)) for h in hmypersons)
    q = f"""SELECT t.HMYPERSON, t.DTMOVEOUT, t.DTLEASETO,
           (SELECT COUNT(*) FROM [{db.DB_YARDI}].dbo.Lease_History lh
            WHERE lh.hTent = t.HMYPERSON AND lh.dtLeaseFrom > '{s}'
              AND lh.sStatus IN ('Scheduled', 'Current')) AS FutureLeases
    FROM [{db.DB_YARDI}].dbo.TENANT t WHERE t.HMYPERSON IN ({ids})"""
    df = db.query(q)
    out: dict[int, str] = {}
    for _, r in df.iterrows():
        hmy = int(r["HMYPERSON"])
        if pd.notna(r["DTMOVEOUT"]):
            o = "transferred" if hmy in transfers else "churned"
        elif (pd.notna(r["DTLEASETO"]) and str(r["DTLEASETO"])[:10] > e) or r["FutureLeases"]:
            o = "renewed"
        else:
            o = "pending"
        out[hmy] = o
    return out


_LINEAGE_CACHE: dict = {}         # (ids-key) -> (ts, {"eff": {...}, "transfer": {...}})
_MOVEIN_TTL = 3600
_TRANSFER_GAP_DAYS = 31


def _build_lineages(property_ids: list[int] | None = None) -> dict:
    """Reconstruct resident lineages across unit transfers (Yardi opens a new TENANT
    record per transfer). Two records chain when they share a property AND either the
    normalized full name OR a non-empty email (Phase 2.3: no stable person key exists in
    TENANT -- hProspect/HAPPLICANT don't link transfers -- so name is primary and email
    is a secondary signal that survives a name change), and the later move-in is within
    31 days of the earlier record's move-out.

    Returns {"eff": {HMYPERSON: effective_movein_date},
             "transfer": set(HMYPERSON whose DTMOVEOUT is an internal transfer, not churn)}.
    Cached 1h."""
    import time as _time
    pids = tuple(sorted(int(p) for p in (property_ids or DEFAULT_PROPERTY_IDS)))
    hit = _LINEAGE_CACHE.get(pids)
    if hit and (_time.time() - hit[0]) < _MOVEIN_TTL:
        return hit[1]
    ids = ",".join(str(p) for p in pids)
    df = db.query(f"""SELECT HMYPERSON,
           UPPER(LTRIM(RTRIM(ISNULL(SFIRSTNAME,''))) + ' ' + LTRIM(RTRIM(ISNULL(SLASTNAME,'')))) AS NM,
           LOWER(LTRIM(RTRIM(ISNULL(SEMAIL,'')))) AS EM,
           HPROPERTY, DTMOVEIN, DTMOVEOUT
        FROM [{db.DB_YARDI}].dbo.TENANT
        WHERE HPROPERTY IN ({ids}) AND DTMOVEIN IS NOT NULL AND {OCC_SQL}""")
    eff: dict[int, date] = {}
    transfer: set[int] = set()
    if df.empty:
        res = {"eff": eff, "transfer": transfer}
        _LINEAGE_CACHE[pids] = (_time.time(), res)
        return res
    df["DTMOVEIN"] = pd.to_datetime(df["DTMOVEIN"], format="mixed", errors="coerce")
    df["DTMOVEOUT"] = pd.to_datetime(df["DTMOVEOUT"], format="mixed", errors="coerce")
    df = df[df["DTMOVEIN"].notna()]
    # A lineage key is (property, name); records with a shared email but different name
    # (e.g. marriage) are merged into the earliest-seen name key for that email+property.
    email_key: dict[tuple, tuple] = {}
    def _key(r):
        base = (int(r.HPROPERTY), r.NM)
        em = r.EM
        if em and "@" in em:
            ek = (int(r.HPROPERTY), em)
            if ek in email_key:
                return email_key[ek]
            email_key[ek] = base
        return base
    df["_k"] = [_key(r) for r in df.itertuples(index=False)]
    for _, g in df.groupby("_k"):
        g = g.sort_values("DTMOVEIN")
        recs = list(g.itertuples(index=False))
        lineage_start = None
        prev_out = None
        prev_hmy = None
        for r in recs:
            chained = (prev_out is not None and pd.notna(prev_out)
                       and 0 <= (r.DTMOVEIN - prev_out).days <= _TRANSFER_GAP_DAYS)
            if not chained:
                lineage_start = r.DTMOVEIN          # new lineage
            elif prev_hmy is not None:
                transfer.add(prev_hmy)              # the PRIOR record's move-out was a transfer
            eff[int(r.HMYPERSON)] = lineage_start.date()
            prev_out, prev_hmy = r.DTMOVEOUT, int(r.HMYPERSON)
    res = {"eff": eff, "transfer": transfer}
    _LINEAGE_CACHE[pids] = (_time.time(), res)
    return res


def effective_moveins(property_ids: list[int] | None = None) -> dict[int, "date"]:
    """HMYPERSON -> effective (transfer-chained) move-in date. See _build_lineages."""
    return _build_lineages(property_ids)["eff"]


def transfer_origin_hmypersons(property_ids: list[int] | None = None) -> set[int]:
    """HMYPERSONs whose DTMOVEOUT is an internal unit TRANSFER (person stayed at the
    property), not a churn -- must not count as churn in training labels or report
    outcomes (Phase 1.1). See _build_lineages."""
    return _build_lineages(property_ids)["transfer"]


def apply_semantic_fixes(df: pd.DataFrame, asof) -> pd.DataFrame:
    """Recompute the features whose SERVE-time semantics diverged from training
    (must be applied IDENTICALLY when building training anchors and when scoring):

      TenureDaysAtAnchor / TenureMonths  -- asof minus person-level EFFECTIVE move-in
        (training SP used current-lease start; serve SP used record move-in: neither is
        residency tenure, and unit transfers truncated it further)
      MoveInYear / MoveInMonth           -- from the effective move-in
      DaysToLeaseEnd                     -- real calendar LeaseToDate - asof, clipped
        [-365, 1825] (serve SP returns 0 by construction at the decision anchor)
      IsLeaseExpiringSoon                -- DaysToLeaseEnd <= 90 (training definition)

    `asof` is a single date/str, or the name of a per-row date column (training passes
    'AnchorDate'). Requires hMyPerson + LeaseToDate columns. Returns a modified copy.

    Phase 0.1: all date parsing uses format="mixed" -- the extraction CSVs mix
    'YYYY-MM-DD' and 'YYYY-MM-DD HH:MM:SS', and pandas' default inference silently NaTs
    the minority format (it once wiped DaysToLeaseEnd for every 2025+ row).
    """
    df = df.copy()
    asof_s = (pd.to_datetime(df[asof], format="mixed", errors="coerce")
              if isinstance(asof, str) and asof in df.columns
              else pd.Series(pd.Timestamp(str(asof)[:10]), index=df.index))
    eff = effective_moveins()
    hmy = pd.to_numeric(df.get("hMyPerson"), errors="coerce")
    eff_in = pd.to_datetime(hmy.map(lambda h: eff.get(int(h)) if pd.notna(h) else None))
    # fall back to the record's own MoveInDate where no chain info exists
    if "MoveInDate" in df.columns:
        eff_in = eff_in.fillna(pd.to_datetime(df["MoveInDate"], format="mixed", errors="coerce"))
    tenure_days = (asof_s - eff_in).dt.days.astype(float)
    df["TenureDaysAtAnchor"] = tenure_days
    df["TenureMonths"] = tenure_days / 30.44
    df["MoveInYear"] = eff_in.dt.year.astype(float)
    df["MoveInMonth"] = eff_in.dt.month.astype(float)
    # renewal-cycle count from CORRECTED tenure (SP derives it from record move-in)
    df["DecisionNumber"] = np.maximum(1.0, np.floor(df["TenureMonths"].fillna(0) / 12.0))
    # canonical property name from PropertyId (serve SP labels 473 'Unknown'; training
    # and serving must encode the same class)
    if "PropertyId" in df.columns:
        pid = pd.to_numeric(df["PropertyId"], errors="coerce")
        named = pid.map(lambda x: PROPERTIES.get(int(x)) if pd.notna(x) else None)
        df["PropertyName"] = named.fillna(df.get("PropertyName"))
    lease_to = pd.to_datetime(df.get("LeaseToDate"), format="mixed", errors="coerce")
    d2e = (lease_to - asof_s).dt.days.clip(lower=-365, upper=1825)
    df["DaysToLeaseEnd"] = d2e.astype(float)
    df["IsLeaseExpiringSoon"] = (d2e <= 90).astype(float)
    if "LeaseFromDate" in df.columns:         # planned term from the same PIT lease
        lease_from = pd.to_datetime(df["LeaseFromDate"], format="mixed", errors="coerce")
        planned = (lease_to - lease_from).dt.days
        prev = pd.to_numeric(df.get("PlannedLeaseDays"), errors="coerce")
        df["PlannedLeaseDays"] = planned.fillna(prev)
    if "PlannedLeaseDays" in df.columns:      # cap corrupt MTM placeholders (train parity)
        df["PlannedLeaseDays"] = pd.to_numeric(
            df["PlannedLeaseDays"], errors="coerce").clip(lower=0, upper=3650)
    # HasLeaseRecord (Phase 2.2): 1 if a Lease_History row anchored the lease fields.
    # The PIT SQL supplies it directly; if absent (legacy frames), infer from LeaseToDate.
    if "HasLeaseRecord" not in df.columns:
        df["HasLeaseRecord"] = lease_to.notna().astype(float)
    else:
        df["HasLeaseRecord"] = pd.to_numeric(df["HasLeaseRecord"], errors="coerce").fillna(0.0)
    # NaT-guard (Phase 0.1): a whole date column silently failing to parse is the bug
    # this phase exists to kill -- fail loud rather than score everyone on medians.
    if len(df) and lease_to.notna().any() and d2e.notna().sum() == 0:
        raise ValueError("apply_semantic_fixes: DaysToLeaseEnd all-NaT despite non-null "
                         "LeaseToDate -- date-format parse failure (see Phase 0.1)")
    return df


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
    ids = ",".join(str(int(p)) for p in property_ids) if property_ids else ",".join(str(p) for p in DEFAULT_PROPERTY_IDS)
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
          AND t.HPROPERTY IN ({ids}) AND t.{OCC_SQL}
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
