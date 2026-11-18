"""
Build a point-in-time TRAINING dataset for the renewal-decision churn model.

One sample = (resident, upcoming-lease-end, month-end anchor). For each
historical month-end anchor A:
  cohort  = residents active at A whose POINT-IN-TIME lease (latest
            Lease_History row with dtLeaseFrom < A) ends in
            (A + MIN_LEAD days, A + MAX_LEAD days]  -- the renewal-candidate
            window the monthly report scores.
  features = same PIT feature block as pit_features.py (18-month WO/memo
            windows ending at A, anchor-time lease attributes, no
            notice/move-out leakage).
  label    = ChurnedAtLeaseEnd: 1 if TENANT.DTMOVEOUT falls within
            [dtLeaseTo - 30d, dtLeaseTo + 90d]; 0 if the resident stayed
            (follow-on lease, extended TENANT.DTLEASETO, or held over with
            no move-out in the window).  Early terminations
            (moved out > 30d before lease end -- possible because move-out
            can happen AFTER the anchor but before the lease end) are
            EXCLUDED, as are right-censored samples (label not yet
            observable).

KEY DIFFERENCE vs sp_GetAnchorSnapshot training data: the cohort comes from
Lease_History, so residents who later RENEWED are present at historical
anchors (the snapshot SP keys on current TENANT.DTLEASEFROM and silently
drops them -- the documented cohort-selection bias).  This dataset IS the
deployment population.

Also adds PIT-safe rent/renewal-history features not in the v4 set:
  PriorLeaseCount, RentCAGRSinceMoveIn, FirstRent, RentVsFirstPct,
  DaysSinceLastRenewal.

READ-ONLY: pure SELECTs (table variables / CTEs only, no writes).

Usage:
    python pit_training.py                       # default anchor range
    python pit_training.py --start 2021-06-30 --end 2026-04-30
"""
from __future__ import annotations

import argparse
import calendar
import os
import sys
import time
from datetime import date

import pandas as pd

from ._cfg import DATA_DIR, get_connection

MIN_LEAD = 45     # days: lease must end at least this far after the anchor
MAX_LEAD = 210    # days: ... and at most this far (renewal-candidate window)
LABEL_BUFFER = 90  # days after lease end needed to observe the label

TRAIN_SQL = """
SET NOCOUNT ON;
DECLARE @AnchorDate DATE = ?;
DECLARE @MinLead INT = ?;
DECLARE @MaxLead INT = ?;
DECLARE @WindowMonths INT = 18;
DECLARE @WindowStartD DATE = DATEADD(MONTH, -@WindowMonths, @AnchorDate);

;WITH base_tenants AS (
    SELECT
        t.HMYPERSON, t.SCODE, t.SUNITCODE, t.HPROPERTY, t.HUNIT,
        t.DTMOVEIN, t.DTMOVEOUT, t.DTLEASEFROM, t.DTLEASETO,
        t.DTNOTICE, t.iNSFCount, t.BNOPAYMENTS, t.SRENT, t.ISTATUS,
        @AnchorDate AS DecisionDate
    FROM Yardi7_Reporting.dbo.TENANT t
    WHERE t.HPROPERTY IN (62, 72, 82, 26)
      AND t.SUNITCODE IS NOT NULL AND t.SUNITCODE NOT LIKE 'WAIT%'
      AND t.DTMOVEIN IS NOT NULL
      AND t.DTMOVEIN <= @AnchorDate
      AND (t.DTMOVEOUT IS NULL OR t.DTMOVEOUT > @AnchorDate)
),
-- point-in-time active lease (latest starting strictly before the anchor)
active_lease AS (
    SELECT
        bt.HMYPERSON,
        lh.cRent       AS ActiveRent,
        lh.iTerm       AS LeaseTermMonths,
        lh.dtLeaseFrom AS ActiveLeaseFrom,
        lh.dtLeaseTo   AS ActiveLeaseTo,
        ROW_NUMBER() OVER (PARTITION BY bt.HMYPERSON ORDER BY lh.dtLeaseFrom DESC) AS rn
    FROM base_tenants bt
    INNER JOIN Yardi7_Reporting.dbo.Lease_History lh
        ON lh.hTent = bt.HMYPERSON
       AND lh.cRent > 0
       AND lh.dtLeaseFrom < bt.DecisionDate
       AND lh.sStatus IN ('Current', 'Past')  -- exclude unchosen Approved/Denied term options (v8.1)
),
active_lease_final AS (
    SELECT * FROM active_lease WHERE rn = 1
),
-- renewal-candidate cohort at this anchor
cohort AS (
    SELECT bt.*, al.ActiveRent, al.LeaseTermMonths, al.ActiveLeaseFrom, al.ActiveLeaseTo
    FROM base_tenants bt
    INNER JOIN active_lease_final al ON al.HMYPERSON = bt.HMYPERSON
    WHERE al.ActiveLeaseTo >  DATEADD(DAY, @MinLead, bt.DecisionDate)
      AND al.ActiveLeaseTo <= DATEADD(DAY, @MaxLead, bt.DecisionDate)
),
base_with_window AS (
    SELECT c.*,
        @WindowStartD AS WindowStart,
        CASE WHEN DATEDIFF(MONTH, c.DTMOVEIN, c.DecisionDate) < 12 THEN 1
             ELSE DATEDIFF(MONTH, c.DTMOVEIN, c.DecisionDate) / 12
        END AS DecisionNumber
    FROM cohort c
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
        SELECT pp.hMyPerson,
               COUNT(*) OVER (PARTITION BY pp.hMyPerson) AS PetCount,
               MAX(CASE WHEN pp.dPetWeight > 50 THEN 1 ELSE 0 END)
                   OVER (PARTITION BY pp.hMyPerson) AS HasLargePets,
               pp.sPetType AS PrimaryPetType,
               ROW_NUMBER() OVER (PARTITION BY pp.hMyPerson ORDER BY pp.sPetType) AS rn
        FROM Yardi7_Reporting.dbo.Person_Pet pp
        WHERE pp.hMyPerson IS NOT NULL AND pp.hMyPerson != 0
    ) ranked WHERE rn = 1
),
prior_lease AS (
    SELECT bw.HMYPERSON,
           lh.cRent AS PriorRent,
           lh.dtLeaseFrom AS PriorLeaseFrom,
           ROW_NUMBER() OVER (PARTITION BY bw.HMYPERSON ORDER BY lh.dtLeaseFrom DESC) AS rn
    FROM base_with_window bw
    INNER JOIN Yardi7_Reporting.dbo.Lease_History lh
        ON lh.hTent = bw.HMYPERSON AND lh.cRent > 0
       AND lh.dtLeaseFrom < bw.ActiveLeaseFrom
       AND lh.sStatus IN ('Current', 'Past')  -- exclude unchosen Approved/Denied term options (v8.1)
),
prior_lease_final AS (
    SELECT HMYPERSON, PriorRent, PriorLeaseFrom FROM prior_lease WHERE rn = 1
),
-- PIT lease-history aggregates (renewal history / rent trajectory)
lease_hist AS (
    SELECT bw.HMYPERSON,
           COUNT(*) AS PriorLeaseCount,
           MIN(CASE WHEN lh.cRent > 0 THEN lh.cRent END) AS MinRentEver
    FROM base_with_window bw
    INNER JOIN Yardi7_Reporting.dbo.Lease_History lh
        ON lh.hTent = bw.HMYPERSON AND lh.dtLeaseFrom < bw.DecisionDate
       AND lh.cRent > 0
       AND lh.sStatus IN ('Current', 'Past')  -- exclude unchosen Approved/Denied term options (v8.1)
    GROUP BY bw.HMYPERSON
),
first_lease AS (
    SELECT HMYPERSON, cRent AS FirstRent FROM (
        SELECT bw.HMYPERSON, lh.cRent,
               ROW_NUMBER() OVER (PARTITION BY bw.HMYPERSON ORDER BY lh.dtLeaseFrom ASC) AS rn
        FROM base_with_window bw
        INNER JOIN Yardi7_Reporting.dbo.Lease_History lh
            ON lh.hTent = bw.HMYPERSON AND lh.cRent > 0
           AND lh.dtLeaseFrom < bw.DecisionDate
           AND lh.sStatus IN ('Current', 'Past')  -- exclude unchosen Approved/Denied term options (v8.1)
    ) x WHERE rn = 1
),
-- follow-on lease (for the LABEL only -- uses post-anchor data by design)
followon AS (
    SELECT bw.HMYPERSON, MIN(lh.dtLeaseFrom) AS NextLeaseFrom
    FROM base_with_window bw
    INNER JOIN Yardi7_Reporting.dbo.Lease_History lh
        ON lh.hTent = bw.HMYPERSON
       AND lh.dtLeaseFrom > bw.ActiveLeaseFrom
       AND lh.dtLeaseFrom <= DATEADD(DAY, 60, bw.ActiveLeaseTo)
       AND lh.cRent > 0
       AND lh.sStatus IN ('Current', 'Past', 'Scheduled')
    GROUP BY bw.HMYPERSON
)
SELECT
    bw.HMYPERSON AS hMyPerson,
    bw.SCODE AS SCODE,
    bw.DecisionNumber,
    bw.DecisionDate,
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
    bw.DTMOVEOUT AS MoveOutDate,
    bw.DTLEASETO AS LeaseToDate,
    bw.ActiveLeaseTo AS AnchorLeaseToDate,
    fo.NextLeaseFrom,

    ISNULL(bw.ActiveRent, bw.SRENT) AS CurrentRent,
    bw.ActiveRent,
    pl.PriorRent,
    CASE WHEN pl.PriorRent IS NOT NULL AND pl.PriorRent > 0
         THEN (bw.ActiveRent - pl.PriorRent) / pl.PriorRent * 100.0 END AS RentIncreasePct,
    bw.LeaseTermMonths,
    bw.ISTATUS AS Status,

    CAST(DATEDIFF(DAY, bw.DTMOVEIN, bw.DecisionDate) AS FLOAT) AS TenureDaysAtAnchor,
    CAST(DATEDIFF(DAY, bw.DTMOVEIN, bw.DecisionDate) / 30.44 AS FLOAT) AS TenureMonths,
    CAST(DATEDIFF(DAY, bw.DecisionDate, bw.ActiveLeaseTo) AS INT) AS DaysToLeaseEnd,
    CAST(DATEDIFF(DAY, bw.ActiveLeaseFrom, bw.ActiveLeaseTo) AS INT) AS PlannedLeaseDays,
    CASE WHEN DATEDIFF(DAY, bw.DecisionDate, bw.ActiveLeaseTo) <= 90 THEN 1 ELSE 0 END AS IsLeaseExpiringSoon,

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
    DATEPART(MONTH, bw.ActiveLeaseTo) AS LeaseToMonth,

    -- new PIT-safe renewal-history features
    ISNULL(lhist.PriorLeaseCount, 1) AS PriorLeaseCount,
    fl.FirstRent,
    CASE WHEN fl.FirstRent > 0
         THEN (bw.ActiveRent - fl.FirstRent) / fl.FirstRent * 100.0 END AS RentVsFirstPct,
    CASE WHEN pl.PriorLeaseFrom IS NOT NULL
         THEN DATEDIFF(DAY, pl.PriorLeaseFrom, bw.DecisionDate) END AS DaysSincePriorLease

FROM base_with_window bw
LEFT JOIN prior_lease_final pl ON pl.HMYPERSON = bw.HMYPERSON
LEFT JOIN wo_agg   wo    ON wo.HMYPERSON   = bw.HMYPERSON
LEFT JOIN memo_agg memo  ON memo.HMYPERSON = bw.HMYPERSON
LEFT JOIN pet_agg  pet   ON pet.hMyPerson  = bw.HMYPERSON
LEFT JOIN lease_hist lhist ON lhist.HMYPERSON = bw.HMYPERSON
LEFT JOIN first_lease fl ON fl.HMYPERSON = bw.HMYPERSON
LEFT JOIN followon fo    ON fo.HMYPERSON = bw.HMYPERSON
ORDER BY bw.HPROPERTY, bw.SUNITCODE;
"""


def month_ends(start: date, end: date):
    out, y, m = [], start.year, start.month
    while True:
        d = date(y, m, calendar.monthrange(y, m)[1])
        if d > end:
            break
        if d >= start:
            out.append(d)
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return out


def label_samples(df: pd.DataFrame, observed_through: pd.Timestamp) -> pd.DataFrame:
    """Attach ChurnedAtLeaseEnd / SampleStatus, exclude unusable rows."""
    df = df.copy()
    le = pd.to_datetime(df['AnchorLeaseToDate'])
    mo = pd.to_datetime(df['MoveOutDate'])
    lt = pd.to_datetime(df['LeaseToDate'])
    nx = pd.to_datetime(df['NextLeaseFrom'])

    churn_lo = le - pd.Timedelta(days=30)
    churn_hi = le + pd.Timedelta(days=90)

    status = pd.Series('Stayed', index=df.index)  # default: held over / renewed
    status[mo.notna() & (mo >= churn_lo) & (mo <= churn_hi)] = 'Churned'
    status[mo.notna() & (mo < churn_lo)] = 'EarlyTermination'
    # right-censored: outcome window not yet fully observable
    status[(churn_hi > observed_through) & (status == 'Stayed')
           & ~(nx.notna() | (lt > churn_hi))] = 'Censored'

    df['SampleStatus'] = status
    df['ChurnedAtLeaseEnd'] = (status == 'Churned').astype(int)
    df['RenewEvidence'] = (nx.notna() | (lt > churn_hi)).astype(int)
    return df


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--start', default='2021-06-30')
    p.add_argument('--end', default='2026-04-30')
    p.add_argument('--out', default=None)
    args = p.parse_args()

    anchors = month_ends(date.fromisoformat(args.start), date.fromisoformat(args.end))
    print(f"[plan] {len(anchors)} month-end anchors: {anchors[0]} .. {anchors[-1]}"
          f"  lead window ({MIN_LEAD}, {MAX_LEAD}] days")

    conn = get_connection()
    cur = conn.cursor()
    frames = []
    for i, a in enumerate(anchors, 1):
        t0 = time.time()
        cur.execute(TRAIN_SQL, a.strftime('%Y-%m-%d'), MIN_LEAD, MAX_LEAD)
        cols = [c[0] for c in cur.description]
        df = pd.DataFrame.from_records(cur.fetchall(), columns=cols)
        df['AnchorDate'] = a.strftime('%Y-%m-%d')
        frames.append(df)
        print(f"  [{i:>2d}/{len(anchors)}] {a}  n={len(df):>4d}  {time.time()-t0:.1f}s")
    conn.close()

    raw = pd.concat(frames, ignore_index=True)
    observed = pd.Timestamp.now().normalize()
    lab = label_samples(raw, observed)

    n0 = len(lab)
    usable = lab[lab['SampleStatus'].isin(['Churned', 'Stayed'])].copy()
    print(f"\n[label] samples={n0}  usable={len(usable)}  "
          f"excluded: early-term={int((lab['SampleStatus']=='EarlyTermination').sum())}"
          f" censored={int((lab['SampleStatus']=='Censored').sum())}")
    print(f"[label] churn rate={usable['ChurnedAtLeaseEnd'].mean():.1%}  "
          f"unique lease-end events="
          f"{usable.groupby(['hMyPerson', 'AnchorLeaseToDate']).ngroups}  "
          f"unique residents={usable['hMyPerson'].nunique()}")
    per_year = usable.groupby(pd.to_datetime(usable['AnchorLeaseToDate']).dt.year)[
        'ChurnedAtLeaseEnd'].agg(['count', 'mean'])
    print(per_year.to_string())

    ts = pd.Timestamp.now().strftime('%Y%m%d_%H%M')
    out = args.out or os.path.join(DATA_DIR, f'pit_renewal_training_{ts}.csv')
    usable.to_csv(out, index=False)
    print(f"\n[OK] wrote {out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
