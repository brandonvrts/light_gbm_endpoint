"""
PIT-safe EXTRA feature families for the renewal-decision model, from the
2026-07-08 read-only DB exploration:

  X_pay_*   resident ledger (TRANS/DETAIL): rent days-late, unpaid, late fees,
            NSF, balances, months-in-arrears, pay-day-of-month, ACH share
  X_mkt_*   market rent at anchor (unit_history 'Rent Change') -> rent gap
  X_prop_*  property occupancy at anchor + trailing move-out rate
  X_unit_*  unit prior-turnover ("churny unit")
  X_ev_*    tenant_history event log: PIT notice (solves DTNOTICE), prior
            renewal / MTM / early-termination / cancel-notice counts
  X_prpsl_* renewal-proposal exposure at anchor (creation-time stamps only)
  X_hh_*    approximate household size (PERSON.dtCreated guard)
  X_sub_*   subsidy/affordable participation (aff_gprhistory) [res-chat 2026-07-09]
  X_conc_*  concession/discount burn-off + ancillary charge-mix (TRANS
            HRETENTIONACCT 30/33/50/51 discounts; 3/17/64/68/71/101 fees)
  X_uatt_*  unit attributes: bathrooms (UNITTYPE.iBaths), floorplan, amenities
            (UNITBUT_AMENITY count/premium/renovation)

Design: ~9 bulk SELECTs (read-only), then all per-anchor point-in-time logic
is vectorized in pandas. Everything is keyed (hMyPerson, AnchorDate) and can
be merged onto any cohort/training frame.

PIT rules honoured (per exploration):
  * never TRANS.SAMOUNTPAID / BOPEN (current-state)
  * receipts count only if SDATEOCCURRED <= anchor
  * proposals via dtProposal/dtApproved/dtSelected/dtDeny <= anchor
    (never sStatus/iInactiveProposal)
  * notice via tenant_history events logged <= anchor, minus Cancel Notice
  * market rent = latest unit_history 'Rent Change' with dtDate <= anchor
  * occupancy excludes never-resident record types (ISTATUS 6,7,8,9)

Usage:
    python extra_features.py                # keys from newest training CSV
    (importable: build_extra_features(keys_df))
"""
from __future__ import annotations

import glob
import os
import sys
import time

import numpy as np
import pandas as pd

from ._cfg import DATA_DIR, get_connection

PROPS = (62, 72, 82, 26)
LATE_THRESH_DAYS = 5
BAL_THRESH = 50.0


def _q(cur, sql, *params):
    cur.execute(sql, *params)
    cols = [c[0] for c in cur.description]
    return pd.DataFrame.from_records(cur.fetchall(), columns=cols)


def _pull_all(cur):
    """All bulk pulls (read-only). Returns dict of raw frames."""
    P = ','.join(str(p) for p in PROPS)
    raw = {}

    t0 = time.time()
    raw['tenant'] = _q(cur, f"""
        SELECT HMYPERSON, HPROPERTY, HUNIT, DTMOVEIN, DTMOVEOUT, ISTATUS
        FROM Yardi7_Reporting.dbo.TENANT WHERE HPROPERTY IN ({P})""")
    print(f"  [pull] TENANT {len(raw['tenant'])} rows {time.time()-t0:.0f}s")

    t0 = time.time()
    raw['rent_charges'] = _q(cur, f"""
        SELECT c.HMY, c.HPERSON, c.SDATEOCCURRED AS due_date,
               MIN(r.SDATEOCCURRED) AS first_pay
        FROM Yardi7_Reporting.dbo.TRANS c
        LEFT JOIN Yardi7_Reporting.dbo.DETAIL d
               ON d.HCHKORCHG = c.HMY AND d.SAMOUNT > 0
        LEFT JOIN Yardi7_Reporting.dbo.TRANS r
               ON r.HMY = d.HINVORREC AND r.ITYPE = 6
        WHERE c.ITYPE = 7 AND c.SNOTES LIKE 'Rent%' AND c.STOTALAMOUNT > 0
          AND c.HPROP IN ({P}) AND c.SDATEOCCURRED >= '2018-01-01'
        GROUP BY c.HMY, c.HPERSON, c.SDATEOCCURRED""")
    print(f"  [pull] rent charges+first-pay {len(raw['rent_charges'])} rows {time.time()-t0:.0f}s")

    t0 = time.time()
    raw['ledger'] = _q(cur, f"""
        SELECT HPERSON, SDATEOCCURRED AS d,
               CASE WHEN ITYPE = 7 THEN STOTALAMOUNT ELSE -STOTALAMOUNT END AS amt
        FROM Yardi7_Reporting.dbo.TRANS
        WHERE ITYPE IN (6, 7) AND HPROP IN ({P}) AND HPERSON > 0
          AND SDATEOCCURRED >= '2015-01-01'""")
    print(f"  [pull] ledger {len(raw['ledger'])} rows {time.time()-t0:.0f}s")

    t0 = time.time()
    raw['latefees'] = _q(cur, f"""
        SELECT HPERSON, SDATEOCCURRED AS d
        FROM Yardi7_Reporting.dbo.TRANS
        WHERE ITYPE = 7 AND SNOTES LIKE '%late%' AND STOTALAMOUNT > 0
          AND HPROP IN ({P}) AND SDATEOCCURRED >= '2018-01-01'""")
    raw['nsf'] = _q(cur, f"""
        SELECT HPERSON, SDATEOCCURRED AS d
        FROM Yardi7_Reporting.dbo.TRANS
        WHERE ITYPE = 6 AND NSF = -1 AND HPROP IN ({P})
          AND SDATEOCCURRED >= '2018-01-01'""")
    raw['receipts'] = _q(cur, f"""
        SELECT HPERSON, SDATEOCCURRED AS d,
               CASE WHEN bACH = -1 THEN 1 ELSE 0 END AS is_ach
        FROM Yardi7_Reporting.dbo.TRANS
        WHERE ITYPE = 6 AND STOTALAMOUNT > 0 AND HPROP IN ({P})
          AND SDATEOCCURRED >= '2018-01-01'""")
    print(f"  [pull] latefees {len(raw['latefees'])} / nsf {len(raw['nsf'])} / "
          f"receipts {len(raw['receipts'])} {time.time()-t0:.0f}s")

    t0 = time.time()
    raw['mkt'] = _q(cur, f"""
        SELECT uh.hUnit, uh.dtDate, uh.cRent
        FROM Yardi7_Reporting.dbo.unit_history uh
        JOIN Yardi7_Reporting.dbo.UNIT u ON u.HMY = uh.hUnit
        WHERE uh.sType = 'Rent Change' AND u.HPROPERTY IN ({P})""")
    raw['unit'] = _q(cur, f"""
        SELECT u.HMY AS hUnit, u.HPROPERTY, u.IBEDROOMS, u.DSQFT, u.EXCLUDE,
               u.HUNITTYPE, ut.iBaths, ut.SDESC AS Floorplan
        FROM Yardi7_Reporting.dbo.UNIT u
        LEFT JOIN Yardi7_Reporting.dbo.UNITTYPE ut ON ut.HMY = u.HUNITTYPE
        WHERE u.HPROPERTY IN ({P})""")
    print(f"  [pull] unit_history {len(raw['mkt'])} / units {len(raw['unit'])} {time.time()-t0:.0f}s")

    t0 = time.time()
    raw['events'] = _q(cur, f"""
        SELECT th.hTent, th.sEvent, th.dtDate, th.dtEventLogged, th.dtNotice
        FROM Yardi7_Reporting.dbo.tenant_history th
        JOIN Yardi7_Reporting.dbo.TENANT t ON t.HMYPERSON = th.hTent
        WHERE t.HPROPERTY IN ({P})
          AND th.sEvent IN ('Notice Given', 'Cancel Notice', 'Lease Renewal',
                            'Month to Month', 'Early Termination', 'Move Out')""")
    print(f"  [pull] tenant_history events {len(raw['events'])} {time.time()-t0:.0f}s")

    t0 = time.time()
    raw['proposals'] = _q(cur, f"""
        SELECT lh.hTent, lh.dtProposal, lh.dtApproved, lh.dtSelected, lh.dtDeny,
               lh.dtLeaseFrom
        FROM Yardi7_Reporting.dbo.Lease_History lh
        JOIN Yardi7_Reporting.dbo.TENANT t ON t.HMYPERSON = lh.hTent
        WHERE t.HPROPERTY IN ({P}) AND lh.dtProposal IS NOT NULL""")
    print(f"  [pull] proposals {len(raw['proposals'])} {time.time()-t0:.0f}s")

    t0 = time.time()
    raw['household'] = _q(cur, f"""
        SELECT r.tenhmy AS hTent,
               rc.code, pe.dtCreated
        FROM Yardi7_Reporting.dbo.roommates r
        CROSS APPLY (VALUES (LTRIM(RTRIM(r.rm1code))), (LTRIM(RTRIM(r.rm2code))),
                            (LTRIM(RTRIM(r.rm3code))), (LTRIM(RTRIM(r.rm4code))),
                            (LTRIM(RTRIM(r.rm5code)))) rc(code)
        LEFT JOIN Yardi7_Reporting.dbo.PERSON pe ON pe.UCODE = rc.code
        JOIN Yardi7_Reporting.dbo.TENANT t ON t.HMYPERSON = r.tenhmy
        WHERE t.HPROPERTY IN ({P}) AND rc.code <> ''""")
    print(f"  [pull] household slots {len(raw['household'])} {time.time()-t0:.0f}s")

    # --- res-chat-derived families (2026-07-09): subsidy, concession, unit attrs ---
    t0 = time.time()
    raw['subsidy'] = _q(cur, f"""
        SELECT g.hTenant, g.dtPeriod, g.cActual_Subsidy_Charge AS sub_charge
        FROM Yardi7_Reporting.dbo.aff_gprhistory g
        JOIN Yardi7_Reporting.dbo.TENANT t ON t.HMYPERSON = g.hTenant
        WHERE t.HPROPERTY IN ({P}) AND g.cActual_Subsidy_Charge <> 0""")
    # discount/concession postings (ITYPE=7): 30 garage-conc, 33 military,
    # 50 senior, 51 employee.  Presence of a posting = concession in effect.
    raw['disc'] = _q(cur, f"""
        SELECT HPERSON, SDATEOCCURRED AS d
        FROM Yardi7_Reporting.dbo.TRANS
        WHERE ITYPE = 7 AND HRETENTIONACCT IN (30, 33, 50, 51)
          AND HPROP IN ({P}) AND SDATEOCCURRED >= '2018-01-01'""")
    # ancillary charge buckets: 3 pet, 17 garage, 64 water, 68 utility,
    # 71 storage, 101 internet -> charge-mix diversity (embeddedness).
    raw['anc'] = _q(cur, f"""
        SELECT HPERSON, SDATEOCCURRED AS d, HRETENTIONACCT AS acct
        FROM Yardi7_Reporting.dbo.TRANS
        WHERE ITYPE = 7 AND HRETENTIONACCT IN (3, 17, 64, 68, 71, 101)
          AND HPROP IN ({P}) AND SDATEOCCURRED >= '2018-01-01'""")
    raw['amenity'] = _q(cur, f"""
        SELECT a.hcode AS hUnit, a.sname, a.CCURRENT AS amt, a.DTCURRENT
        FROM Yardi7_Reporting.dbo.UNITBUT_AMENITY a
        JOIN Yardi7_Reporting.dbo.UNIT u ON u.HMY = a.hcode
        WHERE u.HPROPERTY IN ({P})""")
    print(f"  [pull] subsidy {len(raw['subsidy'])} / disc {len(raw['disc'])} / "
          f"anc {len(raw['anc'])} / amenity {len(raw['amenity'])} {time.time()-t0:.0f}s")

    for k, df in raw.items():
        for c in df.columns:
            if c.lower().startswith('dt') or c in ('d', 'due_date', 'first_pay', 'dtDate'):
                df[c] = pd.to_datetime(df[c], errors='coerce')
    return raw


# ---------------------------------------------------------------------------
def _pay_features(raw, keys):
    rc = raw['rent_charges'].copy()
    out_frames = []
    for anchor, kgrp in keys.groupby('AnchorDate'):
        a = pd.Timestamp(anchor)
        w0 = a - pd.DateOffset(months=12)
        m = rc[(rc['due_date'] > w0) & (rc['due_date'] <= a)].copy()
        paid_by_anchor = m['first_pay'].notna() & (m['first_pay'] <= a)
        days_late = np.where(paid_by_anchor,
                             (m['first_pay'] - m['due_date']).dt.days,
                             (a - m['due_date']).days if False else (a - m['due_date']).dt.days)
        m['days_late'] = np.clip(days_late, 0, None)
        m['unpaid'] = (~paid_by_anchor).astype(int)
        g = m.groupby('HPERSON').agg(
            X_pay_n_rent=('days_late', 'size'),
            X_pay_avg_days_late=('days_late', 'mean'),
            X_pay_max_days_late=('days_late', 'max'),
            X_pay_late5_cnt=('days_late', lambda s: int((s > LATE_THRESH_DAYS).sum())),
            X_pay_unpaid_cnt=('unpaid', 'sum'),
        ).reset_index()

        def _cnt(df, col):
            s = df[(df['d'] > w0) & (df['d'] <= a)].groupby('HPERSON').size()
            return s.rename(col)
        g = g.merge(_cnt(raw['latefees'], 'X_pay_latefee_cnt'), on='HPERSON', how='left')
        g = g.merge(_cnt(raw['nsf'], 'X_pay_nsf_cnt'), on='HPERSON', how='left')

        r = raw['receipts']
        rw = r[(r['d'] > w0) & (r['d'] <= a)]
        pay_day = rw.groupby('HPERSON').agg(
            X_pay_avg_payday=('d', lambda s: float(s.dt.day.mean())),
            X_pay_ach_share=('is_ach', 'mean')).reset_index()
        g = g.merge(pay_day, on='HPERSON', how='left')

        led = raw['ledger']
        bal = led[led['d'] <= a].groupby('HPERSON')['amt'].sum().rename('X_pay_balance')
        g = g.merge(bal, on='HPERSON', how='left')

        # months with balance > $50 in trailing 12 month-ends
        mes = pd.date_range(w0 + pd.DateOffset(months=1), a, freq='ME')
        sub = led[led['d'] <= a].sort_values('d')
        cum = sub.groupby('HPERSON')['amt'].cumsum()
        sub = sub.assign(cum=cum)
        grid = pd.MultiIndex.from_product([g['HPERSON'], mes], names=['HPERSON', 'me']
                                          ).to_frame(index=False)
        grid = grid.sort_values('me')
        sub2 = sub[['HPERSON', 'd', 'cum']].sort_values('d')
        asof = pd.merge_asof(grid, sub2, left_on='me', right_on='d', by='HPERSON')
        arrears = (asof.assign(inarr=(asof['cum'] > BAL_THRESH).astype(int))
                   .groupby('HPERSON')['inarr'].sum().rename('X_pay_months_arrears'))
        g = g.merge(arrears, on='HPERSON', how='left')
        g['AnchorDate'] = anchor
        out_frames.append(g)
    return pd.concat(out_frames, ignore_index=True)


def _mkt_features(raw, keys):
    tan = raw['tenant'][['HMYPERSON', 'HUNIT']]
    mkt = raw['mkt'].sort_values('dtDate')
    unit = raw['unit']
    out = []
    for anchor, kgrp in keys.groupby('AnchorDate'):
        a = pd.Timestamp(anchor)
        cur_m = mkt[mkt['dtDate'] <= a].groupby('hUnit')['cRent'].last().rename('X_mkt_rent')
        old_m = mkt[mkt['dtDate'] <= a - pd.DateOffset(months=12)] \
            .groupby('hUnit')['cRent'].last().rename('mkt_12m_ago')
        g = kgrp[['hMyPerson']].merge(tan, left_on='hMyPerson', right_on='HMYPERSON', how='left')
        g = g.merge(cur_m, left_on='HUNIT', right_index=True, how='left')
        g = g.merge(old_m, left_on='HUNIT', right_index=True, how='left')
        g = g.merge(unit[['hUnit', 'IBEDROOMS', 'DSQFT']],
                    left_on='HUNIT', right_on='hUnit', how='left')
        g['X_mkt_growth_12m'] = (g['X_mkt_rent'] - g['mkt_12m_ago']) / g['mkt_12m_ago'] * 100.0
        g['X_mkt_beds'] = pd.to_numeric(g['IBEDROOMS'], errors='coerce')
        g['X_mkt_sqft'] = pd.to_numeric(g['DSQFT'], errors='coerce')
        g['AnchorDate'] = anchor
        out.append(g[['hMyPerson', 'AnchorDate', 'X_mkt_rent', 'X_mkt_growth_12m',
                      'X_mkt_beds', 'X_mkt_sqft']])
    return pd.concat(out, ignore_index=True)


def _prop_features(raw, keys):
    t = raw['tenant']
    t = t[~t['ISTATUS'].isin([6, 7, 8, 9])]
    units = raw['unit']
    n_units = units[units['EXCLUDE'] == 0].groupby('HPROPERTY')['hUnit'].nunique()
    ev = raw['events']
    mo = ev[ev['sEvent'] == 'Move Out'].merge(
        raw['tenant'][['HMYPERSON', 'HPROPERTY']], left_on='hTent',
        right_on='HMYPERSON', how='left')
    out = []
    for anchor, kgrp in keys.groupby('AnchorDate'):
        a = pd.Timestamp(anchor)
        occ_mask = (t['DTMOVEIN'] <= a) & (t['DTMOVEOUT'].isna() | (t['DTMOVEOUT'] > a))
        occ = t[occ_mask].groupby('HPROPERTY')['HUNIT'].nunique()
        occ_rate = (occ / n_units).rename('X_prop_occupancy')
        mo12 = mo[(mo['dtDate'] > a - pd.DateOffset(months=12)) & (mo['dtDate'] <= a)] \
            .groupby('HPROPERTY').size()
        mo_rate = (mo12 / n_units).rename('X_prop_moveout_rate_12m')
        g = kgrp[['hMyPerson']].merge(
            raw['tenant'][['HMYPERSON', 'HPROPERTY']].drop_duplicates('HMYPERSON'),
            left_on='hMyPerson', right_on='HMYPERSON', how='left')
        g = g.merge(occ_rate, left_on='HPROPERTY', right_index=True, how='left')
        g = g.merge(mo_rate, left_on='HPROPERTY', right_index=True, how='left')
        g['AnchorDate'] = anchor
        out.append(g[['hMyPerson', 'AnchorDate', 'X_prop_occupancy', 'X_prop_moveout_rate_12m']])
    return pd.concat(out, ignore_index=True)


def _unit_features(raw, keys):
    t = raw['tenant']
    t = t[~t['ISTATUS'].isin([6, 7, 8, 9])]
    tan = raw['tenant'][['HMYPERSON', 'HUNIT']].drop_duplicates('HMYPERSON')
    out = []
    for anchor, kgrp in keys.groupby('AnchorDate'):
        a = pd.Timestamp(anchor)
        m = t[(t['DTMOVEIN'] >= '2015-01-01') & (t['DTMOVEIN'] <= a)]
        turns = m.groupby('HUNIT')['DTMOVEIN'] \
            .apply(lambda s: s.dt.normalize().nunique()).rename('X_unit_prior_turns')
        g = kgrp[['hMyPerson']].merge(tan, left_on='hMyPerson', right_on='HMYPERSON', how='left')
        g = g.merge(turns, left_on='HUNIT', right_index=True, how='left')
        g['AnchorDate'] = anchor
        out.append(g[['hMyPerson', 'AnchorDate', 'X_unit_prior_turns']])
    return pd.concat(out, ignore_index=True)


def _event_features(raw, keys):
    ev = raw['events'].copy()
    ev['when'] = ev['dtEventLogged'].fillna(ev['dtDate'])
    out = []
    for anchor, kgrp in keys.groupby('AnchorDate'):
        a = pd.Timestamp(anchor)
        past = ev[ev['when'] <= a]
        g = kgrp[['hMyPerson']].copy()

        notice = past[past['sEvent'] == 'Notice Given'].groupby('hTent')['when'].max()
        cancel = past[past['sEvent'] == 'Cancel Notice'].groupby('hTent')['when'].max()
        nt = pd.DataFrame({'last_notice': notice}).join(
            pd.DataFrame({'last_cancel': cancel}), how='left')
        nt['active'] = nt['last_cancel'].isna() | (nt['last_cancel'] < nt['last_notice'])
        g = g.merge(nt, left_on='hMyPerson', right_index=True, how='left')
        g['X_ev_notice_active'] = (g['active'] == True).astype(int)   # noqa: E712
        g['X_ev_days_since_notice'] = (a - g['last_notice']).dt.days

        for evname, col in [('Lease Renewal', 'X_ev_renewal_cnt'),
                            ('Month to Month', 'X_ev_mtm_cnt'),
                            ('Cancel Notice', 'X_ev_cancelnotice_cnt')]:
            cnt = past[past['sEvent'] == evname].groupby('hTent').size().rename(col)
            g = g.merge(cnt, left_on='hMyPerson', right_index=True, how='left')
        g['AnchorDate'] = anchor
        out.append(g[['hMyPerson', 'AnchorDate', 'X_ev_notice_active',
                      'X_ev_days_since_notice', 'X_ev_renewal_cnt', 'X_ev_mtm_cnt',
                      'X_ev_cancelnotice_cnt']])
    return pd.concat(out, ignore_index=True)


def _proposal_features(raw, keys):
    pr = raw['proposals']
    out = []
    for anchor, kgrp in keys.groupby('AnchorDate'):
        a = pd.Timestamp(anchor)
        m = pr[(pr['dtProposal'] <= a) & (pr['dtProposal'] > a - pd.Timedelta(days=240))]
        g = kgrp[['hMyPerson']].copy()
        agg = m.groupby('hTent').agg(
            last_prpsl=('dtProposal', 'max'),
            selected=('dtSelected', lambda s: int((s.notna() & (s <= a)).any())),
        )
        g = g.merge(agg, left_on='hMyPerson', right_index=True, how='left')
        g['X_prpsl_exists'] = g['last_prpsl'].notna().astype(int)
        g['X_prpsl_days_since'] = (a - g['last_prpsl']).dt.days
        g['X_prpsl_selected'] = g['selected'].fillna(0).astype(int)
        g['AnchorDate'] = anchor
        out.append(g[['hMyPerson', 'AnchorDate', 'X_prpsl_exists',
                      'X_prpsl_days_since', 'X_prpsl_selected']])
    return pd.concat(out, ignore_index=True)


def _hh_features(raw, keys):
    hh = raw['household']
    out = []
    for anchor, kgrp in keys.groupby('AnchorDate'):
        a = pd.Timestamp(anchor)
        m = hh[hh['dtCreated'].isna() | (hh['dtCreated'] <= a)]
        cnt = m.groupby('hTent').size().rename('slots')
        g = kgrp[['hMyPerson']].merge(cnt, left_on='hMyPerson', right_index=True, how='left')
        g['X_hh_size'] = 1 + g['slots'].fillna(0)
        g['AnchorDate'] = anchor
        out.append(g[['hMyPerson', 'AnchorDate', 'X_hh_size']])
    return pd.concat(out, ignore_index=True)


# --- res-chat-derived families (2026-07-09) --------------------------------
def _sub_features(raw, keys):
    """Subsidy / affordable-housing participation from aff_gprhistory.
    Subsidized residents face high relocation friction -> lower churn.
    PIT: only periods with dtPeriod <= anchor."""
    s = raw['subsidy']
    out = []
    for anchor, kgrp in keys.groupby('AnchorDate'):
        a = pd.Timestamp(anchor)
        w0 = a - pd.DateOffset(months=12)
        past = s[s['dtPeriod'] <= a]
        w = past[past['dtPeriod'] > w0]
        ever = past.groupby('hTenant').size().rename('_ever')
        agg = w.groupby('hTenant').agg(
            X_sub_months_12m=('dtPeriod', 'size'),
            X_sub_avg_charge_12m=('sub_charge', 'mean')).reset_index()
        g = kgrp[['hMyPerson']].copy()
        g = g.merge(agg, left_on='hMyPerson', right_on='hTenant', how='left')
        g = g.merge(ever, left_on='hMyPerson', right_index=True, how='left')
        g['X_sub_ever'] = g['_ever'].notna().astype(int)
        g['X_sub_active_12m'] = (g['X_sub_months_12m'].fillna(0) > 0).astype(int)
        g['X_sub_months_12m'] = g['X_sub_months_12m'].fillna(0)
        g['X_sub_avg_charge_12m'] = g['X_sub_avg_charge_12m'].fillna(0.0)
        g['AnchorDate'] = anchor
        out.append(g[['hMyPerson', 'AnchorDate', 'X_sub_ever', 'X_sub_active_12m',
                      'X_sub_months_12m', 'X_sub_avg_charge_12m']])
    return pd.concat(out, ignore_index=True)


def _conc_features(raw, keys):
    """Concession/discount burn-off + ancillary charge-mix from TRANS.
    X_conc_burnoff (had a concession in last 12mo but none in last 3mo) is the
    classic effective-rent-spike churn trigger.  PIT: postings dated <= anchor."""
    disc = raw['disc']
    anc = raw['anc']
    out = []
    for anchor, kgrp in keys.groupby('AnchorDate'):
        a = pd.Timestamp(anchor)
        w12 = a - pd.DateOffset(months=12)
        w3 = a - pd.DateOffset(months=3)
        d12 = disc[(disc['d'] > w12) & (disc['d'] <= a)]
        d3 = disc[(disc['d'] > w3) & (disc['d'] <= a)]
        months12 = (d12.assign(m=d12['d'].dt.to_period('M'))
                    .groupby('HPERSON')['m'].nunique().rename('X_conc_months_12m'))
        a12 = d12.groupby('HPERSON').size().rename('_a12')
        a3 = d3.groupby('HPERSON').size().rename('_a3')
        an = anc[(anc['d'] > w12) & (anc['d'] <= a)]
        anc_types = an.groupby('HPERSON')['acct'].nunique().rename('X_conc_ancillary_types_12m')
        g = kgrp[['hMyPerson']].copy()
        for ser in (months12, a12, a3, anc_types):
            g = g.merge(ser, left_on='hMyPerson', right_index=True, how='left')
        g['X_conc_active_12m'] = (g['_a12'].fillna(0) > 0).astype(int)
        g['X_conc_active_3m'] = (g['_a3'].fillna(0) > 0).astype(int)
        g['X_conc_burnoff'] = ((g['_a12'].fillna(0) > 0) & (g['_a3'].fillna(0) == 0)).astype(int)
        g['X_conc_months_12m'] = g['X_conc_months_12m'].fillna(0)
        g['X_conc_ancillary_types_12m'] = g['X_conc_ancillary_types_12m'].fillna(0)
        g['AnchorDate'] = anchor
        out.append(g[['hMyPerson', 'AnchorDate', 'X_conc_active_12m', 'X_conc_active_3m',
                      'X_conc_burnoff', 'X_conc_months_12m', 'X_conc_ancillary_types_12m']])
    return pd.concat(out, ignore_index=True)


_RENO_KW = ('remodel', 'renovat', 'upgrade', 'diamond', 'palette', 'euro', 'vaulted')


def _uatt_features(raw, keys):
    """Unit attributes: bathrooms (UNITTYPE.iBaths), floorplan (label-encoded,
    leakage-free), and amenities (UNITBUT_AMENITY: count, premium $, renovation
    flag).  Amenities are PIT via DTCURRENT <= anchor; baths/floorplan are
    static.  Unit mapped from the resident's current HUNIT (mirrors the existing
    X_mkt_beds/sqft convention)."""
    tan = raw['tenant'][['HMYPERSON', 'HUNIT']].drop_duplicates('HMYPERSON')
    unit = raw['unit'][['hUnit', 'iBaths', 'Floorplan']].copy()
    fps = sorted(unit['Floorplan'].dropna().astype(str).unique())
    fp_map = {name: i + 1 for i, name in enumerate(fps)}
    unit['X_uatt_floorplan_code'] = unit['Floorplan'].astype(str).map(fp_map).fillna(0).astype(int)
    unit['X_uatt_baths'] = pd.to_numeric(unit['iBaths'], errors='coerce')
    am = raw['amenity'].copy()
    am = am[~am['sname'].isin(['Rent', 'Market'])]
    am['amt'] = pd.to_numeric(am['amt'], errors='coerce').fillna(0.0)
    am['is_reno'] = am['sname'].apply(
        lambda s: int(any(k in s.lower() for k in _RENO_KW)) if isinstance(s, str) else 0)
    out = []
    for anchor, kgrp in keys.groupby('AnchorDate'):
        a = pd.Timestamp(anchor)
        ama = am[am['DTCURRENT'].isna() | (am['DTCURRENT'] <= a)]
        amg = ama.groupby('hUnit').agg(
            X_uatt_amenity_count=('sname', 'size'),
            X_uatt_amenity_premium=('amt', 'sum'),
            X_uatt_has_renovation=('is_reno', 'max')).reset_index()
        g = kgrp[['hMyPerson']].merge(tan, left_on='hMyPerson', right_on='HMYPERSON', how='left')
        g = g.merge(unit[['hUnit', 'X_uatt_baths', 'X_uatt_floorplan_code']],
                    left_on='HUNIT', right_on='hUnit', how='left')
        g = g.merge(amg, on='hUnit', how='left')
        for c, fill in [('X_uatt_amenity_count', 0), ('X_uatt_amenity_premium', 0.0),
                        ('X_uatt_has_renovation', 0), ('X_uatt_floorplan_code', 0)]:
            g[c] = g[c].fillna(fill)
        g['AnchorDate'] = anchor
        out.append(g[['hMyPerson', 'AnchorDate', 'X_uatt_baths', 'X_uatt_floorplan_code',
                      'X_uatt_amenity_count', 'X_uatt_amenity_premium', 'X_uatt_has_renovation']])
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------------------
def load_raw() -> dict:
    """Run the bulk pulls and normalise dtypes. The frames are not anchor-filtered
    (every PIT cut happens in the family functions) and the family functions never
    mutate them, so serving can cache one result and reuse it across requests and
    anchors (light_gbm_endpoint renewal/serve.py, daily cache)."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        raw = _pull_all(cur)
    finally:
        conn.close()
    for k in raw:
        for c in ('HPERSON', 'HMYPERSON', 'hTent', 'hUnit', 'HUNIT', 'hTenant'):
            if c in raw[k].columns:
                raw[k][c] = pd.to_numeric(raw[k][c], errors='coerce').fillna(0).astype('int64')
        for c in raw[k].columns:
            if raw[k][c].dtype == object and c not in ('sEvent', 'code', 'sname', 'Floorplan'):
                try:
                    raw[k][c] = pd.to_numeric(raw[k][c])
                except (ValueError, TypeError):
                    pass
    return raw


def build_extra_features(keys: pd.DataFrame, raw: dict | None = None) -> pd.DataFrame:
    """keys: DataFrame with hMyPerson (int) and AnchorDate (str/date).
    raw: optional pre-pulled `load_raw()` result (serving cache); pulled fresh if None."""
    keys = keys[['hMyPerson', 'AnchorDate']].drop_duplicates().copy()
    keys['hMyPerson'] = pd.to_numeric(keys['hMyPerson']).astype('int64')
    keys['AnchorDate'] = pd.to_datetime(keys['AnchorDate']).dt.strftime('%Y-%m-%d')
    if raw is None:
        raw = load_raw()

    fams = []
    for name, fn in [('pay', _pay_features), ('mkt', _mkt_features),
                     ('prop', _prop_features), ('unit', _unit_features),
                     ('ev', _event_features), ('prpsl', _proposal_features),
                     ('hh', _hh_features), ('sub', _sub_features),
                     ('conc', _conc_features), ('uatt', _uatt_features)]:
        t0 = time.time()
        f = fn(raw, keys)
        if 'HPERSON' in f.columns:
            f = f.rename(columns={'HPERSON': 'hMyPerson'})
        f['hMyPerson'] = pd.to_numeric(f['hMyPerson']).astype('int64')
        f = f.drop_duplicates(subset=['hMyPerson', 'AnchorDate'])
        keep = ['hMyPerson', 'AnchorDate'] + [c for c in f.columns if c.startswith('X_')]
        fams.append(f[keep])
        print(f"  [family] {name}: {len(f)} rows, {len(keep)-2} features, {time.time()-t0:.0f}s")

    out = keys.copy()
    for f in fams:
        out = out.merge(f, on=['hMyPerson', 'AnchorDate'], how='left')
    return out


def main():
    csvs = sorted(glob.glob(os.path.join(DATA_DIR, 'pit_renewal_training_*.csv')))
    if not csvs:
        print('[ERROR] no pit_renewal_training CSV yet'); return 1
    base = pd.read_csv(csvs[-1])
    print(f"[keys] {csvs[-1]}  ({len(base)} rows)")
    feats = build_extra_features(base[['hMyPerson', 'AnchorDate']])
    ts = pd.Timestamp.now().strftime('%Y%m%d_%H%M')
    out = os.path.join(DATA_DIR, f'pit_extra_features_{ts}.csv')
    feats.to_csv(out, index=False)
    ncov = {c: f"{feats[c].notna().mean():.0%}" for c in feats.columns if c.startswith('X_')}
    print(f"[coverage] {ncov}")
    print(f"[OK] {out}  ({len(feats)} rows, {sum(c.startswith('X_') for c in feats.columns)} features)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
