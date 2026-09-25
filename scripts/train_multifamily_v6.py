"""Train multifamily churn v6 -- the roadmap Phase 0-2 retrain (2026-07-15).

Builds on v5-person-tenure. New in v6:
  1.1 TRANSFER-AWARE LABELS. A move-out that is an internal unit transfer (person stays
      at the property, data.transfer_origin_hmypersons) is NOT churn -> relabel to 0.
  1.2 SPLINE CALIBRATION. calibration.SplineCalibrator (monotone PCHIP through
      isotonic-smoothed per-logit-bin rates) replaces the single global sigmoid: smooth
      AND flexible, so mid-range under-prediction and floor-clumping both drop. Floor 0.005.
  1.4 OUT-OF-TIME HOLDOUT. The last 6 anchor-months are reserved entirely (no group
      shuffle) for reported metrics -> honest, non-optimistic AUC/calibration.
  1.3 CAPACITY-BASED TIERS. High tier sized to the property team's monthly outreach
      capacity (default ~top 22% of an expiring cohort); realized recall recorded.
  2.1/2.2/2.4 shared point-in-time SQL (data.pit_features_sql): train/serve parity by
      construction, LEFT JOIN coverage + HasLeaseRecord, curated memo categories.
  3.4 h180 via --horizon 180 (budgeting/occupancy variant).

Run: PYTHONPATH=. .venv/bin/python scripts/train_multifamily_v6.py [--horizon 60|180]
     [--reuse-csv] [--promote]
Honors GBME_BUNDLE_VERSION / GBME_NO_PROMOTE (retrain_pipeline.py gate).
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_curve, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core import db                                            # noqa: E402
from contracts.multifamily_churn import data, prepare_v4       # noqa: E402
from calibration import SplineCalibrator                       # noqa: E402

FEATURES = list(data.MODEL_FEATURES)          # 46 cols incl. HasLeaseRecord
TARGET = "MoveOutInHorizon"
GROUP = "hMyPerson"
MONOTONE_DECREASING = {"TenureDaysAtAnchor", "TenureMonths"}
PROB_FLOOR, PROB_CEIL = 0.005, 0.95
OOT_MONTHS = 6
CAPACITY_FRAC = float(os.environ.get("GBME_HIGH_CAPACITY_FRAC", "0.22"))
WATCH_FRAC = 0.45
DATA_DIR = ROOT / "data"


def _version(horizon: int) -> str:
    return f"v6-roadmap-h{horizon}"


# ---------------------------------------------------------------- extraction
def extract_anchors(start: str, end: str, horizon: int, reuse: bool) -> pd.DataFrame:
    tag = f"anchor_snapshots_v6_h{horizon}"
    if reuse:
        old = sorted(glob.glob(str(DATA_DIR / f"{tag}_*.csv")))
        if old:
            print(f"[extract] reusing {old[-1]}")
            return pd.read_csv(old[-1])
    anchors = pd.date_range(start=start, end=end, freq="ME")
    parts = []
    for i, a in enumerate(anchors, 1):
        d = db.query(data.pit_features_sql(a.date().isoformat(), horizon=horizon))
        if not d.empty:
            parts.append(d)
        print(f"[extract] {a.date()} ({i}/{len(anchors)}): n={len(d)} "
              f"pos={int(d[TARGET].sum()) if not d.empty else 0}", flush=True)
    df = pd.concat(parts, ignore_index=True)
    DATA_DIR.mkdir(exist_ok=True)
    out = DATA_DIR / f"{tag}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    df.to_csv(out, index=False)
    print(f"[extract] {len(df)} samples -> {out}")
    return df


# ---------------------------------------------------------------- preparation
def prepare(df: pd.DataFrame):
    df = df.copy()
    for c in ("AnchorDate", "MoveInDate", "MoveOutDate", "LeaseFromDate", "LeaseToDate"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], format="mixed", errors="coerce")
    df = data.apply_semantic_fixes(df, "AnchorDate")

    # Phase 1.1: relabel transfer move-outs as non-churn.
    transfers = data.transfer_origin_hmypersons()
    is_transfer = df[GROUP].astype("Int64").isin(transfers) & (df[TARGET] == 1)
    n_relabel = int(is_transfer.sum())
    df.loc[is_transfer, TARGET] = 0
    print(f"[label] relabeled {n_relabel} transfer move-outs as non-churn "
          f"({n_relabel / max(1, len(df)):.2%} of samples)")

    yrs = (df["TenureDaysAtAnchor"] / 365.25).clip(lower=0)
    buckets = pd.cut(yrs, [0, 1, 2, 4, 7, 12, 100],
                     labels=["<1y", "1-2y", "2-4y", "4-7y", "7-12y", "12y+"])
    rates = df.groupby(buckets, observed=True)[TARGET].agg(["mean", "count"])
    print("\n[tenure] churn rate by corrected residency tenure:")
    print(rates.rename(columns={"mean": "churn_rate", "count": "n"}).to_string())
    decreasing = rates["mean"].iloc[-2:].mean() < rates["mean"].iloc[:2].mean()
    print(f"[tenure] direction {'DECREASING -> monotone prior applied' if decreasing else 'NOT decreasing -> skipped'}")
    print(f"[coverage] HasLeaseRecord=1 share: {df['HasLeaseRecord'].mean():.1%}")

    prop_enc = LabelEncoder()
    df["PropertyName"] = df["PropertyName"].fillna("Unknown").astype(str)
    prop_enc.fit(list(df["PropertyName"].unique()) + ["Unknown"])
    df["PropertyName_encoded"] = prop_enc.transform(df["PropertyName"])
    pet_enc = LabelEncoder()
    df["PrimaryPetType"] = df["PrimaryPetType"].fillna("None").astype(str)
    pet_enc.fit(list(df["PrimaryPetType"].unique()) + ["None"])
    df["PrimaryPetType_encoded"] = pet_enc.transform(df["PrimaryPetType"])
    for c in ("HasLargePets", "NoPaymentsFlag", "HasNotice", "HasLeaseRecord"):
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


def walkforward(X, y, groups, df_sorted, decreasing, step_months=3):
    dates = pd.to_datetime(df_sorted["AnchorDate"])
    folds = []
    cutoffs = pd.date_range(start="2023-01-01",
                            end=dates.max() - pd.DateOffset(months=step_months),
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
        folds.append({"cutoff": cutoff.strftime("%Y-%m"), "auc": float(auc)})
        print(f"  wf {cutoff.strftime('%Y-%m')}: train={int(tr.sum())} test={int(te.sum())} AUC={auc:.3f}")
    aucs = [f["auc"] for f in folds]
    return {"folds": folds, "mean_auc": float(np.mean(aucs)) if aucs else None,
            "std_auc": float(np.std(aucs)) if aucs else None, "n_folds": len(folds)}


def fit_cohort_intercept(p: np.ndarray, y: np.ndarray) -> float:
    """Additive logit shift a s.t. mean(sigmoid(logit(p)+a)) == mean(y) (bisection)."""
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    target = float(np.mean(y))
    lo, hi = -8.0, 8.0
    for _ in range(60):
        a = (lo + hi) / 2
        m = float(np.mean(1 / (1 + np.exp(-(z + a)))))
        lo, hi = (a, hi) if m < target else (lo, a)
    return (lo + hi) / 2


def capacity_tiers(p: np.ndarray, y: np.ndarray):
    """Phase 1.3: high cut = score at the (1-CAPACITY_FRAC) quantile of the expiring
    cohort (so ~CAPACITY_FRAC of a monthly cohort lands in 'high' = the outreach list);
    watch cut at (1-WATCH_FRAC). Returns (low_max, med_max, policy-with-realized-recall)."""
    high_cut = float(np.quantile(p, 1 - CAPACITY_FRAC))
    med_cut = float(np.quantile(p, 1 - WATCH_FRAC))
    if med_cut >= high_cut:
        med_cut = max(PROB_FLOOR, high_cut - 0.05)
    hi = p >= high_cut
    md = (p >= med_cut) & (~hi)
    tot = max(1, int(y.sum()))
    policy = {
        "capacity_frac": CAPACITY_FRAC, "watch_frac": WATCH_FRAC,
        "high_share": float(hi.mean()), "high_realized_churn": float(y[hi].mean()) if hi.any() else None,
        "high_recall": float(y[hi].sum() / tot),
        "watch_share": float(md.mean()), "watch_realized_churn": float(y[md].mean()) if md.any() else None,
        "high_plus_watch_recall": float(y[hi | md].sum() / tot),
        "basis": "OOT expiring-soon cohort (DaysToLeaseEnd in [-15,60])",
    }
    return med_cut, high_cut, policy


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2022-01-31")
    ap.add_argument("--end", default=None, help="last anchor; default = latest fully-observed")
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--reuse-csv", action="store_true")
    ap.add_argument("--promote", action="store_true")
    args = ap.parse_args()
    version = os.environ.get("GBME_BUNDLE_VERSION", _version(args.horizon))
    no_promote = os.environ.get("GBME_NO_PROMOTE") == "1"
    bundle_dir = ROOT / "bundles" / "multifamily_churn" / version

    # default end = latest anchor whose horizon-day label is fully observable today
    if args.end:
        end = args.end
    else:
        end = (pd.Timestamp.today().normalize() - pd.Timedelta(days=args.horizon)
               - pd.offsets.MonthEnd(1)).date().isoformat()
    print(f"[cfg] version={version} horizon={args.horizon} anchors {args.start}..{end} "
          f"floor={PROB_FLOOR} capacity_frac={CAPACITY_FRAC}")

    raw = extract_anchors(args.start, end, args.horizon, args.reuse_csv)
    X, y, groups, df_sorted, prop_enc, pet_enc, medians, decreasing = prepare(raw)
    print(f"\n[data] {len(X)} samples, {groups.nunique()} tenants, base rate {y.mean():.2%}, "
          f"{df_sorted['AnchorDate'].nunique()} anchors")

    wf = walkforward(X, y, groups, df_sorted, decreasing)
    print(f"[eval] walk-forward AUC {wf['mean_auc']:.3f} +/- {wf['std_auc']:.3f} ({wf['n_folds']} folds)")

    # ---- Phase 1.4: reserve the last OOT_MONTHS anchor-months entirely ----
    amonth = pd.to_datetime(df_sorted["AnchorDate"]).dt.to_period("M")
    oot_periods = sorted(amonth.unique())[-OOT_MONTHS:]
    oot = amonth.isin(oot_periods).values
    print(f"[oot] holdout months {oot_periods[0]}..{oot_periods[-1]}: {int(oot.sum())} samples")

    scaler = StandardScaler().fit(X[~oot])
    Xs = scaler.transform(X)
    # base/calibration split WITHIN the non-OOT window, grouped by person
    dev_idx = np.where(~oot)[0]
    gtr, gcal = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
                     .split(Xs[dev_idx], y.iloc[dev_idx], groups.iloc[dev_idx]))
    tr_idx, cal_idx = dev_idx[gtr], dev_idx[gcal]
    base = build_model(decreasing)
    base.fit(Xs[tr_idx], y.iloc[tr_idx])

    # Phase 1.2: spline calibrator on the calibration split
    calib = SplineCalibrator.fit(base, Xs[cal_idx], y.iloc[cal_idx].values,
                                 floor=PROB_FLOOR, ceil=PROB_CEIL)
    p_cal = calib.predict_proba(Xs[cal_idx])[:, 1]
    print(f"[cal] spline holdout: AUC {roc_auc_score(y.iloc[cal_idx], p_cal):.3f} "
          f"Brier {brier_score_loss(y.iloc[cal_idx], p_cal):.4f} "
          f"distinct {pd.Series(np.round(p_cal,4)).nunique()}")

    # cohort intercept: fit on RECENT (>=2025) non-OOT expiring-soon; validate on OOT
    d2e = X["DaysToLeaseEnd"].values
    adt = pd.to_datetime(df_sorted["AnchorDate"]).values
    fit_mask = (np.isin(np.arange(len(X)), cal_idx) & (d2e >= -15) & (d2e <= 60)
                & (adt >= np.datetime64("2025-01-01")))
    p_fit = calib.predict_proba(Xs[fit_mask])[:, 1]
    a = fit_cohort_intercept(p_fit, y.values[fit_mask])
    final = calib.with_intercept(a)
    print(f"[recal] intercept a={a:.3f} on n={int(fit_mask.sum())} "
          f"(realized {y.values[fit_mask].mean():.2%})")

    # ---- honest OOT expiring-soon evaluation ----
    oot_exp = oot & (d2e >= -15) & (d2e <= 60)
    p_oot = final.predict_proba(Xs[oot_exp])[:, 1]
    y_oot = y.values[oot_exp]
    holdout = {"months": [str(p) for p in oot_periods], "subpop": "OOT DaysToLeaseEnd in [-15,60]",
               "n": int(oot_exp.sum()), "realized_rate": float(y_oot.mean()),
               "predicted_rate": float(p_oot.mean()),
               "auc": float(roc_auc_score(y_oot, p_oot)) if len(set(y_oot)) > 1 else None,
               "pr_auc": float(average_precision_score(y_oot, p_oot)) if len(set(y_oot)) > 1 else None,
               "brier": float(brier_score_loss(y_oot, p_oot)),
               "distinct_scores": int(pd.Series(np.round(p_oot, 4)).nunique())}
    print(f"[oot-eval] n={holdout['n']} realized {holdout['realized_rate']:.2%} "
          f"predicted {holdout['predicted_rate']:.2%} AUC {holdout['auc']} "
          f"Brier {holdout['brier']:.4f} distinct {holdout['distinct_scores']}")

    low_max, med_max, tier_policy = capacity_tiers(p_oot, y_oot)
    thr = float(np.quantile(p_oot, 1 - CAPACITY_FRAC))
    print(f"[tiers] low<{low_max:.3f} watch<{med_max:.3f} high>= | "
          f"high recall {tier_policy['high_recall']:.1%} @ churn {tier_policy['high_realized_churn']}")

    # Kathleen spot-check through the serve path
    bundle_like = {"model": final, "scaler": scaler, "property_encoder": prop_enc,
                   "pet_encoder": pet_enc, "feature_columns": FEATURES, "feature_medians": medians}
    kdf = data.apply_semantic_fixes(data.cohort_features_asof(["t0003779"], "2026-07-14"), "2026-07-14")
    if not kdf.empty:
        kp = final.predict_proba(scaler.transform(prepare_v4.prepare(kdf, bundle_like)))[:, 1]
        print(f"[spot] t0003779 (Kathleen Gray, 14y) -> {kp[0]:.3f}")

    imp = base.feature_importances_.astype(float)
    imp = imp / imp.sum() if imp.sum() else imp
    rank = sorted(zip(FEATURES, imp), key=lambda t: -t[1])
    print("\n[importance] top 12:")
    for f, v in rank[:12]:
        print(f"  {f:26} {v:.4f}")
    print("  tenure ranks:", {f: i + 1 for i, (f, _) in enumerate(rank) if f in MONOTONE_DECREASING})

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    bundle = {
        "model": final, "scaler": scaler, "property_encoder": prop_enc, "pet_encoder": pet_enc,
        "feature_columns": FEATURES, "target_column": TARGET, "feature_medians": medians,
        "version": version, "trained_at": ts, "horizon_days": args.horizon,
        "algorithm": "lgbm", "feature_semantics": "v6-roadmap",
        "monotone_constraints": sorted(MONOTONE_DECREASING) if decreasing else [],
        "transfer_aware_labels": True,
        "population_filter": f"TENANT.{data.OCC_SQL} -- canceled/denied application records "
                             "excluded from anchors and serving (2026-09-25 audit)",
        "tier_labels": {"high": "Priority outreach", "medium": "Watch", "low": "Stable"},
        "tier_policy": tier_policy,
        "data_source": f"data.pit_features_sql monthly anchors {args.start}..{end} h{args.horizon} "
                       "(person-tenure, PIT lease/rent, LEFT-JOIN coverage, curated memos)",
        "train_window": {"n_samples": int(len(X)), "n_tenants": int(groups.nunique()),
                         "churn_rate": float(y.mean()), "anchor_min": args.start, "anchor_max": end},
        "thresholds": {"churn_decision": thr, "risk_low_max": low_max, "risk_medium_max": med_max,
                       "source": "capacity-based (OOT expiring cohort quantiles)"},
        "calibration": {"calibrated": True, "method": "spline (PCHIP/isotonic) + expiring-cohort intercept",
                        "rescale_a": a, "prob_floor": PROB_FLOOR, "prob_ceil": PROB_CEIL},
        "metrics": {"walkforward": wf, "oot_expiring_holdout": holdout},
        "feature_importance": [{"Feature": f, "Importance": float(v)} for f, v in rank],
    }
    bundle_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, bundle_dir / "bundle.joblib")
    with open(bundle_dir / "manifest.json", "w") as fh:
        json.dump({k: v for k, v in bundle.items()
                   if k not in ("model", "scaler", "property_encoder", "pet_encoder")},
                  fh, indent=2, default=str)
    print(f"\n[save] {bundle_dir}/bundle.joblib")
    if args.promote and not no_promote:
        (bundle_dir.parent / "default.txt").write_text(version + "\n")
        print(f"[promote] default.txt -> {version}")
    else:
        print("[promote] skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
