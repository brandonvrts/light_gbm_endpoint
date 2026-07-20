"""DEPRECATED (2026-07-15): superseded by scripts/train_multifamily_v6.py. Kept only to
reproduce the v5-person-tenure bundle. Its inline `_anchor_sql` is a FROZEN historical
copy -- the live train+serve parity path is data.pit_features_sql (v6). Do not extend
this file; edit v6 instead.

Train multifamily churn v5 ("person-tenure") -- the locally-vendored trainer the
retrain pipeline always intended (see scripts/retrain_pipeline.py CONTRACTS comment).

Fixes over v4-anchor-recal (all verified against live data 2026-07-14):
  1. TENURE SEMANTICS. Training defined TenureDaysAtAnchor as anchor - current-lease
     START (0-18 months); serving fed anchor - record move-in (years, out of training
     distribution); unit transfers truncated it further (new TENANT record per transfer,
     ~127 current residents). v5 trains AND serves on person-level EFFECTIVE move-in
     (data.effective_moveins chain) via data.apply_semantic_fixes.
  2. DEGENERATE SERVE FEATURES. Serve SP returns DaysToLeaseEnd=0 / IsLeaseExpiringSoon=1
     by construction (decision-anchored); v5 recomputes both from real calendar dates
     with the training definitions (d2e = LeaseTo - asof, expiring = d2e <= 90).
  3. SNAPPED PROBABILITIES. Isotonic calibration is piecewise-constant (scores pile on
     plateau values like 0.47/1.0) and the serve layer capped display at 0.75. v5 uses
     SIGMOID (smooth) calibration + a cohort-level logit rescale (PlattRescaler) fit on
     REALIZED monthly lease-expiration outcomes (Feb-May 2026), validated on June 2026.
     The serve display squash is retired with this bundle.
  4. TENURE PRIOR. Monotone-decreasing constraint on TenureDaysAtAnchor/TenureMonths:
     longer residency must never raise predicted churn (business prior; direction is
     verified empirically against the training data before applying).

Run:  PYTHONPATH=. .venv/bin/python scripts/train_multifamily_v5.py [--reuse-csv] [--promote]
Bundle -> bundles/multifamily_churn/v5-person-tenure/ (default.txt untouched unless --promote).
"""
from __future__ import annotations
import argparse
import glob
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss, precision_recall_curve, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core import db                                            # noqa: E402
from contracts.multifamily_churn import data, prepare_v4       # noqa: E402
from platt_rescaler import PlattRescaler                       # noqa: E402

FEATURES = list(data.MODEL_FEATURES)          # same 45 columns, corrected semantics
TARGET = "MoveOutInHorizon"
GROUP = "hMyPerson"
MONOTONE_DECREASING = {"TenureDaysAtAnchor", "TenureMonths"}
PROB_FLOOR, PROB_CEIL = 0.02, 0.95
VERSION = "v5-person-tenure"
BUNDLE_DIR = ROOT / "bundles" / "multifamily_churn" / VERSION
DATA_DIR = ROOT / "data"


# ---------------------------------------------------------------- extraction
def _anchor_sql(anchor: str, horizon: int, window_months: int = 18) -> str:
    """Corrected point-in-time anchor snapshot (read-only port of sp_GetAnchorSnapshot).

    Why not the SP: it reads TENANT.DTLEASEFROM/DTLEASETO, which Yardi OVERWRITES on
    renewal -- historical anchors silently drop every resident who later renewed
    (mutated lease-from > anchor fails eligibility) and mislabel lease timing for the
    rest. That made old anchors churner-heavy (fake 8%->2% base-rate decline) and lease
    fields quasi-labels. Here eligibility uses only stable columns (DTMOVEIN/DTMOVEOUT,
    set once) and all lease timing/rent comes from Lease_History as of the anchor.
    WO/memo/pet aggregates are the SP's, verbatim (event logs -- no mutation).
    Scope adds 473 (CCV Townhomes) so its residents train with a real property class.
    `anchor` is a validated ISO date; ints are ints (injection-safe)."""
    a, h, w = anchor, int(horizon), int(window_months)
    ws = (pd.Timestamp(a) - pd.DateOffset(months=w)).date().isoformat()
    he = (pd.Timestamp(a) + pd.Timedelta(days=h)).date().isoformat()
    Y = f"[{db.DB_YARDI}].dbo"
    sql = f"""
    WITH base_tenants AS (
        SELECT t.HMYPERSON, t.SCODE, t.SUNITCODE, t.HPROPERTY, t.HUNIT,
               t.DTMOVEIN, t.DTMOVEOUT, t.DTNOTICE, t.iNSFCount, t.BNOPAYMENTS
        FROM {Y}.TENANT t
        WHERE t.DTMOVEIN IS NOT NULL AND t.DTMOVEIN <= @A
          AND (t.DTMOVEOUT IS NULL OR t.DTMOVEOUT > @A)
          AND t.HPROPERTY IN (62, 72, 82, 26, 473)
          AND t.HUNIT IN (SELECT HMY FROM {Y}.UNIT WHERE EXCLUDE = 0)
    ),
    lease_pit AS (          -- the lease actually in force at @A (immutable history)
        SELECT bt.HMYPERSON, lh.dtLeaseFrom, lh.dtLeaseTo, lh.iTerm,
               ROW_NUMBER() OVER (PARTITION BY bt.HMYPERSON
                                  ORDER BY lh.dtLeaseFrom DESC, lh.dtOccurred DESC,
                                           lh.hMy DESC) AS rn
        FROM base_tenants bt
        JOIN {Y}.Lease_History lh ON lh.hTent = bt.HMYPERSON
        WHERE lh.sStatus IN ('Past', 'Current') AND lh.dtLeaseFrom <= @A
    ),
    lease_pit_final AS (SELECT * FROM lease_pit WHERE rn = 1),
    rent_pit AS (           -- latest priced lease at @A (rent rows only, mirrors SP)
        SELECT bt.HMYPERSON, lh.cRent AS ActiveRent, lh.dtLeaseFrom AS ActiveLeaseFrom,
               ROW_NUMBER() OVER (PARTITION BY bt.HMYPERSON
                                  ORDER BY lh.dtLeaseFrom DESC) AS rn
        FROM base_tenants bt
        JOIN {Y}.Lease_History lh ON lh.hTent = bt.HMYPERSON
        WHERE lh.cRent > 0 AND lh.dtLeaseFrom <= @A AND lh.sStatus IN ('Past', 'Current')
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
            AVG(CAST(CASE WHEN wo.dtwcompl IS NOT NULL AND wo.dtwcompl <= @A
                          THEN DATEDIFF(DAY, wo.dtcall, wo.dtwcompl) END AS FLOAT)) AS AvgDaysToComplete,
            SUM(CASE WHEN wo.dtcall >= DATEADD(DAY, -90,  @A) THEN 1 ELSE 0 END) AS WorkOrders_Last90Days,
            SUM(CASE WHEN wo.dtcall >= DATEADD(DAY, -180, @A) THEN 1 ELSE 0 END) AS WorkOrders_Last180Days,
            SUM(CASE WHEN wo.dtcall >= DATEADD(DAY, -365, @A) THEN 1 ELSE 0 END) AS WorkOrders_Last365Days,
            SUM(CASE WHEN wo.dtcall >= DATEADD(DAY, -90,  @A)
                      AND wo.spriority LIKE '%emergency%' THEN 1 ELSE 0 END) AS EmergencyWO_Last90Days,
            DATEDIFF(DAY, MAX(wo.dtcall), @A) AS DaysSinceLastWorkOrder,
            SUM(CASE WHEN (wo.dtwcompl IS NULL OR wo.dtwcompl > @A)
                      AND wo.dtcall <= @A THEN 1 ELSE 0 END) AS OpenWorkOrdersAtAnchor,
            MAX(CASE WHEN (wo.dtwcompl IS NULL OR wo.dtwcompl > @A) AND wo.dtcall <= @A
                     THEN DATEDIFF(DAY, wo.dtcall, @A) ELSE 0 END) AS MaxOpenWorkOrderAge,
            SUM(CASE WHEN wo.dtcall <= DATEADD(DAY, 30, bt.DTMOVEIN)
                      AND wo.dtcall <= @A THEN 1 ELSE 0 END) AS EarlyWorkOrders30d
        FROM base_tenants bt
        JOIN {Y}.mm2wo wo ON wo.htenant = bt.HMYPERSON
        WHERE wo.dtcall BETWEEN @WS AND @A
        GROUP BY bt.HMYPERSON
    ),
    memo_agg AS (
        SELECT bt.HMYPERSON,
            COUNT(*) AS MemoCount,
            SUM(CASE WHEN m.UDATE >= DATEADD(DAY, -90,  @A) THEN 1 ELSE 0 END) AS Memos_Last90Days,
            SUM(CASE WHEN m.UDATE >= DATEADD(DAY, -180, @A) THEN 1 ELSE 0 END) AS Memos_Last180Days,
            SUM(CASE WHEN m.UDATE >= DATEADD(DAY, -365, @A) THEN 1 ELSE 0 END) AS Memos_Last365Days,
            DATEDIFF(DAY, MAX(m.UDATE), @A) AS DaysSinceLastMemo,
            SUM(CASE WHEN m.STEXT LIKE '%evict%' OR m.STEXT LIKE '%quit%' OR m.STEXT LIKE '%notice%' OR m.STEXT LIKE '%terminate%'
                     THEN 1 ELSE 0 END) AS EvictionMemos,
            SUM(CASE WHEN m.STEXT LIKE '%late%' OR m.STEXT LIKE '%payment%' OR m.STEXT LIKE '%NSF%' OR m.STEXT LIKE '%delinquent%'
                     THEN 1 ELSE 0 END) AS PaymentMemos,
            SUM(CASE WHEN m.STEXT LIKE '%complaint%' OR m.STEXT LIKE '%noise%' OR m.STEXT LIKE '%violation%' OR m.STEXT LIKE '%police%'
                     THEN 1 ELSE 0 END) AS ComplaintMemos,
            SUM(CASE WHEN m.STEXT LIKE '%repair%' OR m.STEXT LIKE '%maintenance%' OR m.STEXT LIKE '%broken%' OR m.STEXT LIKE '%emergency%'
                     THEN 1 ELSE 0 END) AS MaintenanceMemos
        FROM base_tenants bt
        JOIN {Y}.MEMO m ON m.HFILERCD = bt.HMYPERSON
        WHERE m.SSTATUS = 'Memo' AND m.IFILETYPE = '1'
          AND m.UDATE BETWEEN @WS AND @A AND m.STEXT IS NOT NULL
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
        bt.HMYPERSON AS hMyPerson, bt.SCODE, '{a}' AS AnchorDate,
        bt.SUNITCODE AS UnitCode, bt.HPROPERTY AS PropertyId,
        bt.DTMOVEIN AS MoveInDate, bt.DTMOVEOUT AS MoveOutDate,
        lp.dtLeaseFrom AS LeaseFromDate, lp.dtLeaseTo AS LeaseToDate,
        ISNULL(rp.ActiveRent, 0) AS CurrentRent, rp.ActiveRent AS ActiveRent,
        pr.PriorRent AS PriorRent,
        CASE WHEN pr.PriorRent IS NOT NULL AND pr.PriorRent > 0
             THEN (rp.ActiveRent - pr.PriorRent) / pr.PriorRent * 100.0 END AS RentIncreasePct,
        lp.iTerm AS LeaseTermMonths,
        CASE WHEN bt.DTNOTICE IS NOT NULL AND bt.DTNOTICE <= @A
             THEN DATEDIFF(DAY, bt.DTNOTICE, @A) END AS DaysSinceNotice,
        CASE WHEN bt.DTNOTICE IS NOT NULL AND bt.DTNOTICE <= @A THEN 1 ELSE 0 END AS HasNotice,
        ISNULL(CAST(bt.iNSFCount AS INT), 0) AS NSFCount,
        CASE WHEN bt.BNOPAYMENTS = 1 THEN 1 ELSE 0 END AS NoPaymentsFlag,
        CASE WHEN bt.DTMOVEOUT IS NOT NULL AND bt.DTMOVEOUT > @A AND bt.DTMOVEOUT <= @HE
             THEN 1 ELSE 0 END AS MoveOutInHorizon,
        DATEPART(MONTH, @A) AS DecisionMonth, DATEPART(YEAR, @A) AS DecisionYear,
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
    JOIN lease_pit_final lp ON lp.HMYPERSON = bt.HMYPERSON
    LEFT JOIN rent_pit_final rp ON rp.HMYPERSON = bt.HMYPERSON
    LEFT JOIN prior_rent_final pr ON pr.HMYPERSON = bt.HMYPERSON
    LEFT JOIN wo_agg wo ON wo.HMYPERSON = bt.HMYPERSON
    LEFT JOIN memo_agg memo ON memo.HMYPERSON = bt.HMYPERSON
    LEFT JOIN pet_agg pet ON pet.hMyPerson = bt.HMYPERSON"""
    # single result set for db.query: dates as literals, no DECLARE batch
    return (sql.replace("@WS", f"'{ws}'").replace("@HE", f"'{he}'")
               .replace("@A", f"'{a}'"))


def extract_anchors(start: str, end: str, horizon: int, reuse: bool) -> pd.DataFrame:
    if reuse:
        old = sorted(glob.glob(str(DATA_DIR / f"anchor_snapshots_v5pit_h{horizon}_*.csv")))
        if old:
            print(f"[extract] reusing {old[-1]}")
            return pd.read_csv(old[-1])
    anchors = pd.date_range(start=start, end=end, freq="ME")
    parts = []
    for i, a in enumerate(anchors, 1):
        d = db.query(_anchor_sql(a.date().isoformat(), horizon))
        if not d.empty:
            parts.append(d)
        print(f"[extract] {a.date()} ({i}/{len(anchors)}): n={len(d)} "
              f"pos={int(d[TARGET].sum()) if not d.empty else 0}", flush=True)
    df = pd.concat(parts, ignore_index=True)
    DATA_DIR.mkdir(exist_ok=True)
    out = DATA_DIR / (f"anchor_snapshots_v5pit_h{horizon}_"
                      f"{datetime.now().strftime('%Y%m%d_%H%M')}.csv")
    df.to_csv(out, index=False)
    print(f"[extract] {len(df)} samples -> {out}")
    return df


# ---------------------------------------------------------------- preparation
def prepare(df: pd.DataFrame):
    df = df.copy()
    for c in ("AnchorDate", "MoveInDate", "MoveOutDate", "LeaseFromDate", "LeaseToDate"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    df = data.apply_semantic_fixes(df, "AnchorDate")           # THE v5 fix, train side

    # Empirical tenure->churn direction (guards the monotone prior)
    yrs = (df["TenureDaysAtAnchor"] / 365.25).clip(lower=0)
    buckets = pd.cut(yrs, [0, 1, 2, 4, 7, 12, 100],
                     labels=["<1y", "1-2y", "2-4y", "4-7y", "7-12y", "12y+"])
    rates = df.groupby(buckets, observed=True)[TARGET].agg(["mean", "count"])
    print("\n[tenure] churn rate by corrected residency tenure:")
    print(rates.rename(columns={"mean": "churn_rate", "count": "n"}).to_string())
    decreasing = rates["mean"].iloc[-2:].mean() < rates["mean"].iloc[:2].mean()
    print(f"[tenure] direction {'DECREASING -> monotone prior applied' if decreasing else 'NOT decreasing -> prior skipped'}")

    prop_enc = LabelEncoder()
    df["PropertyName"] = df["PropertyName"].fillna("Unknown").astype(str)
    prop_enc.fit(list(df["PropertyName"].unique()) + ["Unknown"])
    df["PropertyName_encoded"] = prop_enc.transform(df["PropertyName"])
    pet_enc = LabelEncoder()
    df["PrimaryPetType"] = df["PrimaryPetType"].fillna("None").astype(str)
    pet_enc.fit(list(df["PrimaryPetType"].unique()) + ["None"])
    df["PrimaryPetType_encoded"] = pet_enc.transform(df["PrimaryPetType"])
    for c in ("HasLargePets", "NoPaymentsFlag", "HasNotice"):
        df[c] = df[c].fillna(0).astype(int)

    medians = {}
    for c in FEATURES:
        if c not in df.columns:
            df[c] = 0.0
        s = pd.to_numeric(df[c], errors="coerce")
        m = float(s.median()) if s.notna().any() else 0.0
        medians[c] = m
        df[c] = s.fillna(m)
    df = df.sort_values("AnchorDate").reset_index(drop=True)
    return (df[FEATURES], df[TARGET].astype(int), df[GROUP], df,
            prop_enc, pet_enc, medians, decreasing)


def build_model(decreasing: bool):
    mono = [-1 if f in MONOTONE_DECREASING and decreasing else 0 for f in FEATURES]
    return LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15,
                          max_depth=6, min_child_samples=50, subsample=0.8,
                          subsample_freq=1, colsample_bytree=0.8,
                          reg_alpha=0.5, reg_lambda=0.5, is_unbalance=True,
                          monotone_constraints=mono,
                          random_state=42, n_jobs=-1, verbose=-1)


# ---------------------------------------------------------------- evaluation
def walkforward(X, y, groups, df_sorted, decreasing, step_months=3):
    dates = pd.to_datetime(df_sorted["AnchorDate"])
    folds = []
    cutoffs = pd.date_range(start="2023-01-01", end=dates.max() - pd.DateOffset(months=step_months),
                            freq=f"{step_months}MS")
    for cutoff in cutoffs:
        te = (dates >= cutoff) & (dates < cutoff + pd.DateOffset(months=step_months))
        tr = (dates < cutoff) & (~groups.isin(set(groups[te])))
        if tr.sum() < 500 or te.sum() < 200 or y[te].sum() < 15:
            continue
        sc = StandardScaler().fit(X[tr])
        m = build_model(decreasing)
        m.fit(sc.transform(X[tr]), y[tr])
        auc = roc_auc_score(y[te], m.predict_proba(sc.transform(X[te]))[:, 1])
        folds.append({"cutoff": cutoff.strftime("%Y-%m"), "train": int(tr.sum()),
                      "test": int(te.sum()), "auc": float(auc)})
        print(f"  wf {cutoff.strftime('%Y-%m')}: train={tr.sum()} test={te.sum()} AUC={auc:.3f}")
    aucs = [f["auc"] for f in folds]
    return {"folds": folds, "mean_auc": float(np.mean(aucs)) if aucs else None,
            "std_auc": float(np.std(aucs)) if aucs else None, "n_folds": len(folds)}


# ------------------------------------------------- cohort recalibration
# NOTE: an earlier design fit the rescale on realized PAST monthly cohorts scored through
# the serve SP. That leaks: the SP anchors features at COALESCE(DTMOVEOUT, DTLEASETO,...),
# so decided tenants carry post-outcome features (notice on file, move-out-anchored
# windows) -> AUC 1.0 and a degenerate step-function rescale (2 distinct scores). The
# honest population-shift fix below uses only TRAINING anchors (point-in-time labels):
# the holdout subpopulation whose lease expires within ~60 days IS the monthly-report
# cohort, and an intercept-only shift matches its realized rate without touching spread.
def fit_cohort_intercept(p: np.ndarray, y: np.ndarray) -> float:
    """Solve a so that mean(sigmoid(logit(p) + a)) == mean(y). Intercept-only (b=1):
    corrects the level for the expiring-cohort population without reshaping ranks."""
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    target = float(np.mean(y))
    lo, hi = -8.0, 8.0
    for _ in range(60):
        a = (lo + hi) / 2
        m = float(np.mean(1 / (1 + np.exp(-(z + a)))))
        lo, hi = (a, hi) if m < target else (lo, a)
    return (lo + hi) / 2


def _max_f1_threshold(y, p):
    pr, rc, th = precision_recall_curve(y, p)
    f1 = np.where((pr + rc) > 0, 2 * pr * rc / (pr + rc), 0)[:-1]
    return float(th[int(np.argmax(f1))]) if len(f1) else 0.5


def derive_buckets(p, y, target_high=0.55, target_med=0.30):
    d = pd.DataFrame({"p": p, "y": y}).sort_values("p", ascending=False)
    d["cum_rate"] = d["y"].cumsum() / np.arange(1, len(d) + 1)
    hi = d[(d["cum_rate"] >= target_high) & (np.arange(1, len(d) + 1) >= 10)]
    high_cut = float(hi["p"].min()) if len(hi) else float(np.quantile(p, 0.9))
    md = d[(d["cum_rate"] >= target_med) & (np.arange(1, len(d) + 1) >= 30)]
    med_cut = float(md["p"].min()) if len(md) else float(np.quantile(p, 0.7))
    if med_cut >= high_cut:
        med_cut = max(0.05, high_cut - 0.10)
    return med_cut, high_cut


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2022-01-31")
    ap.add_argument("--end", default="2026-01-31")   # leaves Feb+ cohorts out-of-sample
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--reuse-csv", action="store_true")
    ap.add_argument("--promote", action="store_true")
    args = ap.parse_args()

    raw = extract_anchors(args.start, args.end, args.horizon, args.reuse_csv)
    X, y, groups, df_sorted, prop_enc, pet_enc, medians, decreasing = prepare(raw)
    print(f"\n[data] {len(X)} samples, {groups.nunique()} tenants, "
          f"base rate {y.mean():.2%}, {df_sorted['AnchorDate'].nunique()} anchors")

    wf = walkforward(X, y, groups, df_sorted, decreasing)
    print(f"[eval] walk-forward AUC {wf['mean_auc']:.3f} +/- {wf['std_auc']:.3f} ({wf['n_folds']} folds)")

    # final: fit base on 80%, sigmoid-calibrate on held-out 20% (grouped by person)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    tr, cal = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
                   .split(Xs, y, groups))
    base = build_model(decreasing)
    base.fit(Xs[tr], y.iloc[tr])
    calibrated = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid")
    calibrated.fit(Xs[cal], y.iloc[cal])
    p_cal = calibrated.predict_proba(Xs[cal])[:, 1]
    print(f"[cal] sigmoid holdout: AUC {roc_auc_score(y.iloc[cal], p_cal):.3f}  "
          f"Brier {brier_score_loss(y.iloc[cal], p_cal):.4f}  "
          f"anchor base rate {y.iloc[cal].mean():.2%}")

    # cohort-level intercept: the monthly report scores the EXPIRING cohort, whose base
    # rate is far above the all-actives anchor rate. Fit intercept-only on the held-out
    # expiring-soon subpopulation (point-in-time labels -- no serve-SP leakage).
    d2e_cal = X["DaysToLeaseEnd"].iloc[cal].values
    anchor_cal = pd.to_datetime(df_sorted["AnchorDate"].iloc[cal]).values
    # RECENT anchors only: portfolio churn fell sharply over the training window
    # (2022 ~7-8% of actives vs 2025 ~2%), so a level fit on all years runs hot for
    # current cohorts. One intercept parameter needs little data.
    exp_mask = ((d2e_cal >= -15) & (d2e_cal <= 60)
                & (anchor_cal >= np.datetime64("2025-01-01")))
    p_exp, y_exp = p_cal[exp_mask], y.iloc[cal][exp_mask].values
    print(f"\n[recal] expiring-soon holdout subpop (anchors >= 2025-01): n={exp_mask.sum()}, "
          f"realized {y_exp.mean():.2%}, pre-shift predicted {p_exp.mean():.2%}")
    a = fit_cohort_intercept(p_exp, y_exp)
    b = 1.0
    final = PlattRescaler(calibrated, a=a, b=b)
    z_exp = np.log(np.clip(p_exp, 1e-6, 1 - 1e-6) / (1 - np.clip(p_exp, 1e-6, 1 - 1e-6)))
    p_post = np.clip(1 / (1 + np.exp(-(z_exp + a))), PROB_FLOOR, PROB_CEIL)
    holdout = {"subpop": "cal-holdout DaysToLeaseEnd in [-15, 60]",
               "n": int(exp_mask.sum()), "realized_rate": float(y_exp.mean()),
               "predicted_rate_post": float(p_post.mean()),
               "auc": float(roc_auc_score(y_exp, p_exp)),
               "brier": float(brier_score_loss(y_exp, p_post)),
               "distinct_scores": int(pd.Series(np.round(p_post, 4)).nunique())}
    print(f"[recal] intercept a={a:.3f} (b=1) -> predicted {holdout['predicted_rate_post']:.2%}, "
          f"AUC {holdout['auc']:.3f}, Brier {holdout['brier']:.4f}, "
          f"distinct scores {holdout['distinct_scores']}")

    # spot-check Kathleen Gray through the full serve path
    bundle_like = {"model": final, "scaler": scaler, "property_encoder": prop_enc,
                   "pet_encoder": pet_enc, "feature_columns": FEATURES,
                   "feature_medians": medians}
    kp = final.predict_proba(scaler.transform(
        prepare_v4.prepare(data.apply_semantic_fixes(
            data.features_by_scodes(["t0003779"], "2026-07-14"), "2026-07-14"),
            bundle_like)))[:, 1]
    print(f"[recal] spot-check t0003779 (Kathleen Gray, 14y tenure) -> {kp[0]:.3f}")

    low_max, med_max = derive_buckets(p_post, y_exp)
    thr = _max_f1_threshold(y_exp, p_post)
    print(f"[thresholds] decision {thr:.3f} | buckets low<{low_max:.2f} med<{med_max:.2f}")

    imp = np.mean([getattr(c.estimator, "estimator", c.estimator).feature_importances_
                   for c in calibrated.calibrated_classifiers_], axis=0).astype(float)
    imp = imp / imp.sum()
    rank = sorted(zip(FEATURES, imp), key=lambda t: -t[1])
    print("\n[importance] top 12:")
    for f, v in rank[:12]:
        print(f"  {f:26} {v:.4f}")
    print("  tenure ranks:", {f: i + 1 for i, (f, _) in enumerate(rank)
                              if f in MONOTONE_DECREASING})

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    bundle = {
        "model": final, "scaler": scaler,
        "property_encoder": prop_enc, "pet_encoder": pet_enc,
        "feature_columns": FEATURES, "target_column": TARGET,
        "feature_medians": medians,
        "version": VERSION, "trained_at": ts, "horizon_days": args.horizon,
        "algorithm": "lgbm", "feature_semantics": "v5-person-tenure",
        "monotone_constraints": sorted(MONOTONE_DECREASING) if decreasing else [],
        "data_source": f"sp_GetAnchorSnapshot monthly anchors {args.start}..{args.end} h{args.horizon}, "
                       "person-tenure + calendar lease-end semantics (data.apply_semantic_fixes)",
        "train_window": {"n_samples": int(len(X)), "n_tenants": int(groups.nunique()),
                         "churn_rate": float(y.mean()),
                         "anchor_min": args.start, "anchor_max": args.end},
        "thresholds": {"churn_decision": thr, "risk_low_max": low_max,
                       "risk_medium_max": med_max,
                       "source": "cohort-outcome max-F1 + cumulative-churn buckets"},
        "calibration": {"calibrated": True,
                        "method": "sigmoid + expiring-cohort intercept (b=1)",
                        "rescale_a": a, "rescale_b": b,
                        "rescale_basis": holdout["subpop"],
                        "prob_floor": PROB_FLOOR, "prob_ceil": PROB_CEIL},
        "metrics": {"walkforward": wf, "expiring_cohort_holdout": holdout},
        "feature_importance": [{"Feature": f, "Importance": float(v)} for f, v in rank],
    }
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, BUNDLE_DIR / "bundle.joblib")
    with open(BUNDLE_DIR / "manifest.json", "w") as fh:
        json.dump({k: v for k, v in bundle.items()
                   if k not in ("model", "scaler", "property_encoder", "pet_encoder")},
                  fh, indent=2, default=str)
    print(f"\n[save] {BUNDLE_DIR}/bundle.joblib")
    if args.promote:
        (BUNDLE_DIR.parent / "default.txt").write_text(VERSION + "\n")
        print(f"[promote] default.txt -> {VERSION}")
    else:
        print("[promote] skipped (pass --promote to flip default.txt)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
