"""
Point-in-time feature extraction for cohort forecasting.

Replicates sp_GetTenantFeaturesByScode's feature computation but with a TRUE
point-in-time anchor: DecisionDate = @AnchorDate for every tenant (the SP's
COALESCE(DTMOVEOUT, DTLEASETO, @AsOfDate) leaks outcomes into the features),
and lease-timing features (DaysToLeaseEnd, PlannedLeaseDays, LeaseToMonth,
IsLeaseExpiringSoon) computed from the Lease_History lease ACTIVE AT THE
ANCHOR rather than the current TENANT row (which reflects post-renewal state).

Everything else mirrors the SPs so the feature distribution matches the v4
training data (sp_GetAnchorSnapshot):
  * 18-month WO/memo windows ending at the anchor
  * active lease = latest Lease_History row with dtLeaseFrom strictly < anchor
  * EarlyWorkOrders30d keys off DTMOVEIN (immutable).  It previously used the
    current TENANT.DTLEASEFROM, which moves forward when a resident renews:
    at a historical anchor the 30-day cutoff then landed AFTER the anchor and
    the feature silently collapsed into WorkOrderCount for renewers only
    (equal on 96.6% of renewers vs 28.7% of churners).  That encoded the
    outcome and inflated walk-forward AUC from ~0.64 to 0.92.  Fixed 2026-11-10.
  * iNSFCount / BNOPAYMENTS are still CURRENT values and are therefore NOT
    point-in-time safe.  Excluded from the modelled feature set; use the
    PIT-safe X_pay_nsf_cnt from extra_features.py instead.
"""
from __future__ import annotations

import pandas as pd

from ._cfg import get_connection

PIT_FEATURE_SQL = """
SET NOCOUNT ON;
DECLARE @AnchorDate DATE = ?;
DECLARE @WindowMonths INT = 18;
DECLARE @WindowStartD DATE = DATEADD(MONTH, -@WindowMonths, @AnchorDate);

DECLARE @scodes TABLE (SCODE VARCHAR(50));
INSERT INTO @scodes (SCODE)
SELECT LTRIM(RTRIM(value)) FROM STRING_SPLIT(?, ',')
WHERE LTRIM(RTRIM(value)) <> '';

;WITH base_tenants AS (
    SELECT
        t.HMYPERSON, t.SCODE, t.SFIRSTNAME, t.SLASTNAME, t.SUNITCODE,
        t.HPROPERTY, t.DTMOVEIN, t.DTMOVEOUT, t.DTLEASEFROM, t.DTLEASETO,
        t.DTRENEWDATE, t.DTNOTICE, t.iNSFCount, t.BNOPAYMENTS, t.SRENT, t.ISTATUS,
        @AnchorDate AS DecisionDate
    FROM Yardi7_Reporting.dbo.TENANT t
    INNER JOIN @scodes s ON LOWER(RTRIM(t.SCODE)) = LOWER(s.SCODE)
    WHERE t.DTMOVEIN IS NOT NULL
      AND t.DTMOVEIN <= @AnchorDate                     -- resident at anchor
      AND (t.DTMOVEOUT IS NULL OR t.DTMOVEOUT > @AnchorDate)
),
base_with_window AS (
    SELECT bt.*,
        @WindowStartD AS WindowStart,
        CASE WHEN DATEDIFF(MONTH, bt.DTMOVEIN, bt.DecisionDate) < 12 THEN 1
             ELSE DATEDIFF(MONTH, bt.DTMOVEIN, bt.DecisionDate) / 12
        END AS DecisionNumber
    FROM base_tenants bt
),
wo_agg AS (
    SELECT
        bw.HMYPERSON,
        COUNT(*) AS WorkOrderCount,
        SUM(CASE WHEN wo.spriority LIKE '%emergency%' THEN 1 ELSE 0 END) AS EmergencyWorkOrders,
        AVG(CAST(CASE WHEN wo.dtwcompl IS NOT NULL AND wo.dtwcompl <= bw.DecisionDate
                      THEN DATEDIFF(DAY, wo.dtcall, wo.dtwcompl) END AS FLOAT)) AS AvgDaysToComplete,
        SUM(CASE WHEN wo.dtcall >= DATEADD(DAY, -90,  bw.DecisionDate) THEN 1 ELSE 0 END) AS WorkOrders_Last90Days,
        SUM(CASE WHEN wo.dtcall >= DATEADD(DAY, -180, bw.DecisionDate) THEN 1 ELSE 0 END) AS WorkOrders_Last180Days,
        SUM(CASE WHEN wo.dtcall >= DATEADD(DAY, -365, bw.DecisionDate) THEN 1 ELSE 0 END) AS WorkOrders_Last365Days,
        SUM(CASE WHEN wo.dtcall >= DATEADD(DAY, -90,  bw.DecisionDate)
                  AND wo.spriority LIKE '%emergency%' THEN 1 ELSE 0 END) AS EmergencyWO_Last90Days,
        DATEDIFF(DAY, MAX(wo.dtcall), bw.DecisionDate) AS DaysSinceLastWorkOrder,
        SUM(CASE WHEN (wo.dtwcompl IS NULL OR wo.dtwcompl > bw.DecisionDate)
                  AND wo.dtcall <= bw.DecisionDate THEN 1 ELSE 0 END) AS OpenWorkOrdersAtAnchor,
        MAX(CASE WHEN (wo.dtwcompl IS NULL OR wo.dtwcompl > bw.DecisionDate)
                  AND wo.dtcall <= bw.DecisionDate
                 THEN DATEDIFF(DAY, wo.dtcall, bw.DecisionDate) ELSE 0 END) AS MaxOpenWorkOrderAge,
        SUM(CASE WHEN wo.dtcall <= DATEADD(DAY, 30, bw.DTMOVEIN)
                  AND wo.dtcall <= bw.DecisionDate THEN 1 ELSE 0 END) AS EarlyWorkOrders30d
    FROM base_with_window bw
    INNER JOIN Yardi7_Reporting.dbo.mm2wo wo ON wo.htenant = bw.HMYPERSON
    WHERE wo.dtcall BETWEEN bw.WindowStart AND bw.DecisionDate
    GROUP BY bw.HMYPERSON, bw.DecisionDate, bw.DTMOVEIN
),
memo_agg AS (
    SELECT
        bw.HMYPERSON,
        COUNT(*) AS MemoCount,
        SUM(CASE WHEN m.UDATE >= DATEADD(DAY, -90,  bw.DecisionDate) THEN 1 ELSE 0 END) AS Memos_Last90Days,
        SUM(CASE WHEN m.UDATE >= DATEADD(DAY, -180, bw.DecisionDate) THEN 1 ELSE 0 END) AS Memos_Last180Days,
        SUM(CASE WHEN m.UDATE >= DATEADD(DAY, -365, bw.DecisionDate) THEN 1 ELSE 0 END) AS Memos_Last365Days,
        DATEDIFF(DAY, MAX(m.UDATE), bw.DecisionDate) AS DaysSinceLastMemo,
        SUM(CASE WHEN m.STEXT LIKE '%evict%' OR m.STEXT LIKE '%quit%' OR m.STEXT LIKE '%notice%' OR m.STEXT LIKE '%terminate%'
                 THEN 1 ELSE 0 END) AS EvictionMemos,
        SUM(CASE WHEN m.STEXT LIKE '%late%' OR m.STEXT LIKE '%payment%' OR m.STEXT LIKE '%NSF%' OR m.STEXT LIKE '%delinquent%'
                 THEN 1 ELSE 0 END) AS PaymentMemos,
        SUM(CASE WHEN m.STEXT LIKE '%complaint%' OR m.STEXT LIKE '%noise%' OR m.STEXT LIKE '%violation%' OR m.STEXT LIKE '%police%'
                 THEN 1 ELSE 0 END) AS ComplaintMemos,
        SUM(CASE WHEN m.STEXT LIKE '%repair%' OR m.STEXT LIKE '%maintenance%' OR m.STEXT LIKE '%broken%' OR m.STEXT LIKE '%emergency%'
                 THEN 1 ELSE 0 END) AS MaintenanceMemos
    FROM base_with_window bw
    INNER JOIN Yardi7_Reporting.dbo.MEMO m ON m.HFILERCD = bw.HMYPERSON
    WHERE m.SSTATUS = 'Memo' AND m.IFILETYPE = '1'
      AND m.UDATE BETWEEN bw.WindowStart AND bw.DecisionDate
      AND m.STEXT IS NOT NULL
    GROUP BY bw.HMYPERSON, bw.DecisionDate
),
pet_agg AS (
    SELECT hMyPerson, PetCount, HasLargePets, PrimaryPetType
    FROM (
        SELECT
            pp.hMyPerson,
            COUNT(*) OVER (PARTITION BY pp.hMyPerson) AS PetCount,
            MAX(CASE WHEN pp.dPetWeight > 50 THEN 1 ELSE 0 END)
                OVER (PARTITION BY pp.hMyPerson) AS HasLargePets,
            pp.sPetType AS PrimaryPetType,
            ROW_NUMBER() OVER (PARTITION BY pp.hMyPerson ORDER BY pp.sPetType) AS rn
        FROM Yardi7_Reporting.dbo.Person_Pet pp
        WHERE pp.hMyPerson IS NOT NULL AND pp.hMyPerson != 0
    ) ranked
    WHERE rn = 1
),
-- Lease ACTIVE AT ANCHOR (strict point-in-time; carries its own dtLeaseTo)
active_lease AS (
    SELECT
        bw.HMYPERSON,
        lh.cRent       AS ActiveRent,
        lh.iTerm       AS LeaseTermMonths,
        lh.dtLeaseFrom AS ActiveLeaseFrom,
        lh.dtLeaseTo   AS ActiveLeaseTo,
        ROW_NUMBER() OVER (PARTITION BY bw.HMYPERSON ORDER BY lh.dtLeaseFrom DESC) AS rn
    FROM base_with_window bw
    INNER JOIN Yardi7_Reporting.dbo.Lease_History lh
        ON lh.hTent = bw.HMYPERSON
       AND lh.cRent > 0
       AND lh.dtLeaseFrom < bw.DecisionDate
),
active_lease_final AS (
    SELECT HMYPERSON, ActiveRent, LeaseTermMonths, ActiveLeaseFrom, ActiveLeaseTo
    FROM active_lease WHERE rn = 1
),
prior_lease AS (
    SELECT
        al.HMYPERSON,
        lh.cRent AS PriorRent,
        ROW_NUMBER() OVER (PARTITION BY al.HMYPERSON ORDER BY lh.dtLeaseFrom DESC) AS rn
    FROM active_lease_final al
    INNER JOIN Yardi7_Reporting.dbo.Lease_History lh
        ON lh.hTent = al.HMYPERSON
       AND lh.cRent > 0
       AND lh.dtLeaseFrom < al.ActiveLeaseFrom
),
prior_lease_final AS (
    SELECT HMYPERSON, PriorRent FROM prior_lease WHERE rn = 1
)
SELECT
    bw.HMYPERSON AS hMyPerson,
    bw.SCODE AS SCODE,
    bw.DecisionNumber,
    bw.WindowStart,
    bw.DecisionDate,
    @WindowMonths AS WindowMonths,
    bw.SFIRSTNAME + ' ' + bw.SLASTNAME AS FullName,
    bw.SUNITCODE AS UnitCode,
    bw.HPROPERTY AS PropertyId,
    CASE bw.HPROPERTY
        WHEN 62 THEN 'Country Club Village Apartments'
        WHEN 72 THEN 'Sierra Pointe'
        WHEN 82 THEN 'Residences at 62W'
        WHEN 26 THEN 'Winwood'
        ELSE 'Unknown'
    END AS PropertyName,
    bw.DTMOVEIN AS MoveInDate,
    bw.DTMOVEOUT AS MoveOutDate,          -- outcome bookkeeping only, NOT a feature
    bw.DTLEASEFROM AS LeaseFromDate,      -- current values, for reference
    bw.DTLEASETO AS LeaseToDate,
    al.ActiveLeaseTo AS AnchorLeaseToDate,  -- lease end as KNOWN AT ANCHOR

    ISNULL(al.ActiveRent, bw.SRENT) AS CurrentRent,
    al.ActiveRent,
    pl.PriorRent,
    CASE WHEN pl.PriorRent IS NOT NULL AND pl.PriorRent > 0
         THEN (al.ActiveRent - pl.PriorRent) / pl.PriorRent * 100.0 END AS RentIncreasePct,
    al.LeaseTermMonths,
    bw.ISTATUS AS Status,

    CAST(DATEDIFF(DAY, bw.DTMOVEIN, bw.DecisionDate) AS FLOAT) AS TenureDaysAtAnchor,
    CAST(DATEDIFF(DAY, bw.DTMOVEIN, bw.DecisionDate) / 30.44 AS FLOAT) AS TenureMonths,

    -- Lease timing from the ANCHOR-TIME lease (not current TENANT row)
    CAST(DATEDIFF(DAY, bw.DecisionDate, al.ActiveLeaseTo) AS INT) AS DaysToLeaseEnd,
    CAST(DATEDIFF(DAY, al.ActiveLeaseFrom, al.ActiveLeaseTo) AS INT) AS PlannedLeaseDays,
    CASE WHEN DATEDIFF(DAY, bw.DecisionDate, al.ActiveLeaseTo) <= 90
         THEN 1 ELSE 0 END AS IsLeaseExpiringSoon,

    CASE WHEN bw.DTNOTICE IS NOT NULL AND bw.DTNOTICE <= bw.DecisionDate
         THEN DATEDIFF(DAY, bw.DTNOTICE, bw.DecisionDate) END AS DaysSinceNotice,
    CASE WHEN bw.DTNOTICE IS NOT NULL AND bw.DTNOTICE <= bw.DecisionDate
         THEN 1 ELSE 0 END AS HasNotice,
    ISNULL(CAST(bw.iNSFCount AS INT), 0) AS NSFCount,
    CASE WHEN bw.BNOPAYMENTS = 1 THEN 1 ELSE 0 END AS NoPaymentsFlag,

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

    CAST(ISNULL(wo.WorkOrderCount, 0) AS FLOAT) / @WindowMonths * 12 AS WorkOrderFrequency,
    CAST(ISNULL(memo.MemoCount, 0) AS FLOAT) / @WindowMonths * 12 AS MemoFrequency,

    ISNULL(pet.PetCount, 0) AS PetCount,
    ISNULL(pet.HasLargePets, 0) AS HasLargePets,
    ISNULL(pet.PrimaryPetType, 'None') AS PrimaryPetType,

    DATEPART(MONTH, bw.DTMOVEIN) AS MoveInMonth,
    DATEPART(YEAR,  bw.DTMOVEIN) AS MoveInYear,
    DATEPART(MONTH, bw.DecisionDate) AS DecisionMonth,
    DATEPART(YEAR,  bw.DecisionDate) AS DecisionYear,
    DATEPART(MONTH, al.ActiveLeaseTo) AS LeaseToMonth

FROM base_with_window bw
LEFT JOIN active_lease_final al ON al.HMYPERSON = bw.HMYPERSON
LEFT JOIN prior_lease_final  pl ON pl.HMYPERSON = bw.HMYPERSON
LEFT JOIN wo_agg   wo   ON wo.HMYPERSON   = bw.HMYPERSON
LEFT JOIN memo_agg memo ON memo.HMYPERSON = bw.HMYPERSON
LEFT JOIN pet_agg  pet  ON pet.hMyPerson  = bw.HMYPERSON
ORDER BY bw.HPROPERTY, bw.SUNITCODE;
"""


def fetch_pit_features(scodes: list[str], anchor_date: str) -> pd.DataFrame:
    """Point-in-time features for `scodes` as of `anchor_date` (YYYY-MM-DD).

    Tenants not resident at the anchor (moved in later, or already moved out)
    are simply absent from the result -- callers reconcile against the input
    list to report non-forecastable residents.
    """
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(PIT_FEATURE_SQL, anchor_date, ','.join(scodes))
        cols = [c[0] for c in cur.description]
        df = pd.DataFrame.from_records(cur.fetchall(), columns=cols)
    finally:
        conn.close()
    for c in ['MoveInDate', 'MoveOutDate', 'LeaseFromDate', 'LeaseToDate',
              'AnchorLeaseToDate', 'DecisionDate']:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors='coerce')
    return df
