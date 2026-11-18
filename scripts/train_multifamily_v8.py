"""Train, validate and bundle v8: the PIT-safe renewal-decision churn model.

Port of multifamily-prediction@feature/mu-fa25-mom-churn-reports (commit 195cceb):
  1. walk-forward validation  -- train_renewal_model.walk_forward, unchanged
  2. isotonic recalibration   -- calibrate_isotonic.crossfit_pool logic, unchanged
  3. final fit on all resolved lease events + production calibrator on the full pool
  4. bundle -> bundles/multifamily_churn/<version>/{bundle.joblib, manifest.json}

Feature set = the branch's validated 44: V4_FEATURES minus NOT_PIT_SAFE
(NSFCount, NoPaymentsFlag) + NOTICE_SAFE (tenant_history notice) + PIT_SAFE_SUBS
(X_pay_nsf_cnt). Same as score_mom_retrained.load_training().

Inputs (build first, read-only against Yardi):
  python -m contracts.multifamily_churn.renewal.pit_training
  python -m contracts.multifamily_churn.renewal.extra_features
Run:
  PYTHONPATH=. .venv/bin/python scripts/train_multifamily_v8.py [--version v8-renewal-pit]
Never touches default.txt -- promotion is a separate, deliberate step.
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
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from contracts.multifamily_churn.renewal import train_renewal_model as T  # noqa: E402
from contracts.multifamily_churn.renewal._cfg import DATA_DIR  # noqa: E402

MIN_FIT, CLIP_LO, CLIP_HI = 1500, 0.01, 0.99     # calibrate_isotonic.py constants
RISK_LOW_MAX, RISK_MEDIUM_MAX = 0.25, 0.45        # score_mom_retrained.RISK_BANDS
LEAD_MIN, LEAD_MAX = 45, 210                      # pit_training MIN_LEAD/MAX_LEAD
TRAINED_PROPERTIES = [62, 72, 82, 26]             # pit_training base_tenants filter


def newest(pattern: str) -> str:
    c = sorted(glob.glob(os.path.join(DATA_DIR, pattern)))
    if not c:
        raise SystemExit(f"[ERROR] no {pattern} in {DATA_DIR}")
    return c[-1]


def model_cols(df: pd.DataFrame) -> list[str]:
    cols = ([c for c in T.V4_FEATURES if c in df.columns and c not in T.NOT_PIT_SAFE]
            + [c for c in T.NOTICE_SAFE if c in df.columns]
            + [c for c in T.PIT_SAFE_SUBS if c in df.columns])
    return list(dict.fromkeys(cols))


def hl_chi2(p, y, k=5):
    d = pd.DataFrame({"p": p, "y": y})
    d["b"] = pd.qcut(d["p"], k, duplicates="drop")
    chi = 0.0
    for _, g in d.groupby("b", observed=True):
        pb, ab = g["p"].mean(), g["y"].mean()
        if 0 < pb < 1:
            chi += len(g) * (pb - ab) ** 2 / (pb * (1 - pb))
    return float(chi)


def calib_report(p, y) -> dict:
    return {"n": int(len(y)), "pred_rate": float(np.mean(p)), "actual_rate": float(np.mean(y)),
            "auc": float(roc_auc_score(y, p)), "brier": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))), "hl_chi2": hl_chi2(p, y)}


def smooth_isotonic(x, y):
    """Isotonic fit -> monotone PCHIP through its block centres, clipped to [CLIP_LO, CLIP_HI]."""
    from scipy.interpolate import PchipInterpolator
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(x, y)
    k = (pd.DataFrame({"x": iso.X_thresholds_, "y": iso.y_thresholds_})
         .groupby("y", as_index=False)["x"].mean().sort_values("x"))
    f = PchipInterpolator(k["x"].values, k["y"].values, extrapolate=False)
    lo_x, hi_x = k["x"].min(), k["x"].max()

    def apply(z):
        z = np.clip(np.asarray(z, float), lo_x, hi_x)
        return np.clip(f(z), CLIP_LO, CLIP_HI)
    return apply


def crossfit_isotonic(pool: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """calibrate_isotonic.crossfit_pool: each anchor quarter calibrated only on lease
    events that resolved before it, never on its own events."""
    pool = pool.copy()
    pool["lease_end"] = pool["AnchorDate"] + pd.to_timedelta(pool["DaysToLeaseEnd"], unit="D")
    out = []
    for q in sorted(pool["quarter"].unique()):
        cur = pool[pool["quarter"] == q]
        fit = pool[pool["lease_end"] < cur["AnchorDate"].min()]
        fit = fit[~fit["event_id"].isin(set(cur["event_id"]))]
        if len(fit) < MIN_FIT:
            continue
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(
            fit[score_col].values, fit["ChurnedAtLeaseEnd"].values)
        c = cur.copy()
        c["p_iso"] = np.clip(iso.predict(cur[score_col].values), CLIP_LO, CLIP_HI)
        c["p_cal"] = smooth_isotonic(fit[score_col].values,
                                     fit["ChurnedAtLeaseEnd"].values)(cur[score_col].values)
        out.append(c)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def band_table(p, y) -> list[dict]:
    base = float(np.mean(y))
    rows = []
    for name, lo, hi in (("low", 0, RISK_LOW_MAX), ("medium", RISK_LOW_MAX, RISK_MEDIUM_MAX),
                         ("high", RISK_MEDIUM_MAX, 1.01)):
        m = (p >= lo) & (p < hi)
        if m.sum():
            rows.append({"band": name, "n": int(m.sum()), "share": float(m.mean()),
                         "actual_churn": float(y[m].mean()), "lift": float(y[m].mean() / base)})
    return rows


def top_k_lift(p, y, frac) -> dict:
    k = max(1, int(round(len(p) * frac)))
    idx = np.argsort(-p)[:k]
    return {"frac": frac, "n": k, "churn": float(y[idx].mean()),
            "lift": float(y[idx].mean() / y.mean()), "movers_captured": int(y[idx].sum()),
            "movers_total": int(y.sum())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v8-renewal-pit")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--extra", default=None)
    args = ap.parse_args()

    csv = args.csv or newest("pit_renewal_training_*.csv")
    extra = args.extra or newest("pit_extra_features_*.csv")
    df = pd.read_csv(csv, low_memory=False)
    xf = pd.read_csv(extra, low_memory=False)
    for d in (df, xf):
        d["hMyPerson"] = pd.to_numeric(d["hMyPerson"]).astype("int64")
    df = df.merge(xf, on=["hMyPerson", "AnchorDate"], how="left")
    df = T.prepare(df)
    cols = model_cols(df)
    print(f"[data] {os.path.basename(csv)} + {os.path.basename(extra)}: samples={len(df)} "
          f"events={df['event_id'].nunique()} residents={df['hMyPerson'].nunique()} "
          f"churn={df['ChurnedAtLeaseEnd'].mean():.1%}  features={len(cols)}")
    if len(cols) != 44:
        print(f"[WARN] expected the branch's 44 features, got {len(cols)}")

    os.environ.setdefault("PYTHONHASHSEED", "0")
    params = T.load_best_params()
    print(f"[params] {params}")

    # ---- 1. walk-forward (branch protocol, unchanged) ----------------------------
    no_pet = [c for c in cols if c not in ("PetCount", "HasLargePets", "PrimaryPetType_encoded")]
    print("\nWalk-forward (quarterly, label-buffered, deduped test):")
    summary, folds, pool = T.walk_forward(df, {"v8": cols, "v8_no_pets": no_pet}, params,
                                          include_v4=False)
    y = pool["ChurnedAtLeaseEnd"].values
    p = pool["p_v8"].values

    # ---- 2. isotonic, cross-fitted out-of-sample ---------------------------------
    cf = crossfit_isotonic(pool, "p_v8")
    yc = cf["ChurnedAtLeaseEnd"].values
    calib = {"raw": calib_report(cf["p_v8"].values, yc),
             "isotonic_step_out_of_sample": calib_report(cf["p_iso"].values, yc),
             "isotonic_smoothed_out_of_sample": calib_report(cf["p_cal"].values, yc)}
    print("\n[calibration, cross-fitted pool]")
    for k, v in calib.items():
        print(f"  {k:24s} n={v['n']} pred={v['pred_rate']:.1%} act={v['actual_rate']:.1%} "
              f"AUC={v['auc']:.4f} Brier={v['brier']:.4f} HL={v['hl_chi2']:.1f}")
    bands = band_table(cf["p_cal"].values, yc)
    lifts = [top_k_lift(p, y, 0.10), top_k_lift(p, y, 0.20)]
    for b in bands:
        print(f"  band {b['band']:6s} n={b['n']:5d} share={b['share']:.0%} "
              f"churn={b['actual_churn']:.1%} lift={b['lift']:.2f}x")
    for l in lifts:
        print(f"  top {l['frac']:.0%}: churn={l['churn']:.1%} lift={l['lift']:.2f}x")

    # ---- 3. production fit -------------------------------------------------------
    fit = df.copy()
    encoders = {}
    for raw, enc in (("PropertyName", "PropertyName_encoded"),
                     ("PrimaryPetType", "PrimaryPetType_encoded")):
        vals = fit[raw].astype(str)
        le = LabelEncoder().fit(vals)
        encoders[raw] = {"classes": [str(c) for c in le.classes_], "fallback": str(vals.mode().iat[0])}
        fit[enc] = le.transform(vals)
    X, med = T.feature_matrix(fit, cols)
    model = LGBMClassifier(**params)
    model.fit(X, fit["ChurnedAtLeaseEnd"], sample_weight=fit["sample_weight"])
    # Production calibrator: isotonic fit on the full pool, then a monotone PCHIP through
    # the isotonic block centres. Deviation from the branch's raw step function: plain
    # isotonic collapses scores into ~200 tied levels (49-resident cohort -> 19 distinct)
    # and costs ranking; on the same cross-fit, smoothed = AUC 0.6337 / Brier 0.1872 /
    # HL 13.4 vs step 0.6309 / 0.1875 / 11.3 (2026-11-12).
    grid_x = np.linspace(0, 1, 201)
    grid_y = smooth_isotonic(p, y)(grid_x)
    imp = pd.Series(model.booster_.feature_importance("gain"), index=cols)
    imp = (imp / imp.sum()).sort_values(ascending=False)

    v8 = summary["v8"]
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    meta = {
        "version": args.version, "trained_at": ts, "algorithm": "lgbm",
        "feature_semantics": "v8-renewal-pit",
        "target_column": "ChurnedAtLeaseEnd",
        "target_definition": "move-out within [lease_end - 30d, lease_end + 90d]; early "
                             "terminations and right-censored events excluded",
        "lead_window_days": [LEAD_MIN, LEAD_MAX],
        "trained_properties": TRAINED_PROPERTIES,
        "horizon_days": None,
        "source": "real-production",
        "lineage": "RealtyRTS/multifamily-prediction@feature/mu-fa25-mom-churn-reports 195cceb (v2/)",
        "data_source": f"{os.path.basename(csv)} + {os.path.basename(extra)}",
        "train_window": {"n_samples": int(len(df)), "n_events": int(df["event_id"].nunique()),
                         "n_tenants": int(df["hMyPerson"].nunique()),
                         "churn_rate": float(df["ChurnedAtLeaseEnd"].mean()),
                         "anchor_min": str(df["AnchorDate"].min().date()),
                         "anchor_max": str(df["AnchorDate"].max().date())},
        "feature_columns": cols,
        "feature_medians": {k: float(v) for k, v in med.items()},
        "params": params,
        "thresholds": {"risk_low_max": RISK_LOW_MAX, "risk_medium_max": RISK_MEDIUM_MAX,
                       "source": "branch score_mom_retrained RISK_BANDS, on calibrated probability"},
        "calibration": {"method": "isotonic + monotone PCHIP smoothing", "fit_rows": int(len(pool)),
                        "clip": [CLIP_LO, CLIP_HI],
                        "grid_x": [round(float(v), 5) for v in grid_x],
                        "grid_y": [round(float(v), 6) for v in grid_y],
                        "crossfit": calib,
                        "prob_floor": CLIP_LO, "prob_ceil": CLIP_HI},
        "metrics": {
            "walkforward": {"mean_auc": v8["fold_mean"], "std_auc": v8["fold_std"],
                            "n_folds": len(folds), "pooled_auc": v8["pooled_auc"],
                            "pooled_auc_ci95": v8["ci"], "undecided_auc": v8["undecided_auc"],
                            "n_undecided": v8["n_undecided"], "pr_auc": v8["pr_auc"],
                            "brier_raw": v8["brier"], "n": v8["n"], "base_rate": v8["base_rate"],
                            "folds": folds},
            "no_pets_check": summary["v8_no_pets"],
            "risk_bands_calibrated": bands, "top_k_lift_raw": lifts,
            "branch_reference": {"pooled_auc": 0.6366, "ci95": [0.623, 0.650],
                                 "undecided_auc": 0.6209, "brier": 0.1901, "n": 8083},
        },
        "feature_importance": [{"Feature": k, "Importance": float(v)} for k, v in imp.items()],
        "usage": "churn_probability = isotonic-calibrated (stated probability); "
                 "model_raw_probability = uncalibrated (rank outreach queues by this).",
    }
    out = ROOT / "bundles" / "multifamily_churn" / args.version
    out.mkdir(parents=True, exist_ok=True)
    bundle = dict(meta, model=model, encoders=encoders, bundle_format="v8-renewal")
    joblib.dump(bundle, out / "bundle.joblib")
    (out / "manifest.json").write_text(json.dumps(meta, indent=2, default=str))
    pool.to_csv(os.path.join(DATA_DIR, f"renewal_wf_pool_v8_{ts}.csv"), index=False)
    print(f"\n[OK] {out}/bundle.joblib  (default.txt NOT changed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
