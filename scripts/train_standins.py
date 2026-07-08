"""Phase 1: train interim stand-in LightGBM bundles from REAL labeled data (SSMS),
with a synthetic-signal fallback. Saves bundles the registry serves until the real
production .joblib is dropped in.

Run: . .venv/bin/activate && python scripts/train_standins.py
"""
from __future__ import annotations
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
import joblib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from contracts.multifamily_churn import data as mf_data            # noqa: E402
from contracts.multifamily_churn import prepare_features as mf_prep  # noqa: E402
from contracts.signal_desk_brokerage import data as bk_data        # noqa: E402
from contracts.signal_desk_brokerage import prepare_features as bk_prep  # noqa: E402
from contracts.signal_desk_brokerage.schema import FEATURES as BK_FEATURES  # noqa: E402

BUNDLES = ROOT / "bundles"
TS = datetime.now().strftime("%Y%m%d_%H%M%S")


def _new_model():
    base = LGBMClassifier(n_estimators=250, num_leaves=31, learning_rate=0.05,
                          is_unbalance=True, min_child_samples=20, n_jobs=4, verbose=-1)
    try:
        return CalibratedClassifierCV(base, method="sigmoid", cv=3)
    except Exception:
        return base


def _fit(X: pd.DataFrame, y: np.ndarray):
    """Honest holdout AUC for the manifest, then refit on all data for the saved model."""
    from sklearn.model_selection import train_test_split
    auc = float("nan")
    try:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=7, stratify=y)
        m = _new_model(); m.fit(Xtr, ytr)
        auc = float(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))
    except Exception as e:
        print("   holdout err:", str(e)[:100])
    model = _new_model(); model.fit(X, y)   # final model trained on everything
    return model, auc


def _save(contract: str, version: str, bundle: dict, manifest: dict):
    d = BUNDLES / contract / version
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, d / "bundle.joblib")
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    (BUNDLES / contract / "default.txt").write_text(version)
    print(f"  saved {contract}/{version}/bundle.joblib  (AUC={manifest.get('train_auc')}, "
          f"source={manifest['source']}, n={manifest['n_train']})")


def _synth(features: list[str], signal_cols: list[str], n: int = 2500):
    rng = np.random.default_rng(7)
    X = pd.DataFrame({c: rng.normal(0, 1, n) for c in features})
    z = sum(X[c] for c in signal_cols if c in X) + rng.normal(0, 1, n)
    y = (z > np.quantile(z, 0.7)).astype(int)
    return X, y


def train_multifamily():
    print("multifamily-churn:")
    scodes = mf_data.scodes_for_properties([62, 72, 82, 26], limit=1200)
    frames = []
    for i in range(0, len(scodes), 150):
        chunk = scodes[i:i + 150]
        try:
            frames.append(mf_data.features_by_scodes(chunk, as_of="2025-06-30"))
        except Exception as e:
            print("   batch err:", str(e)[:100])
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    src, n, auc = "synthetic", 0, float("nan")
    if not df.empty and "ActualChurn" in df:
        y = pd.to_numeric(df["ActualChurn"], errors="coerce")
        keep = y.isin([0, 1])
        df, y = df[keep].reset_index(drop=True), y[keep].astype(int).to_numpy()
    else:
        y = np.array([])
    encoders, medians = (mf_prep.fit_prep(df) if not df.empty else ({}, {}))
    # Stand-in only: exclude near-deterministic notice features so the interim model
    # produces a sensible probability spread (the real v2 bundle uses all 45 + calibration).
    _leak = ("HasNotice", "DaysSinceNotice", "DaysToLeaseEnd", "IsLeaseExpiringSoon",
             "PlannedLeaseDays", "NoPaymentsFlag", "NSFCount")
    model_input = [f for f in mf_data.MODEL_FEATURES if f not in _leak]
    if len(y) >= 150 and len(set(y)) == 2:
        X = mf_prep.prepare(df, encoders, medians)
        model, auc = _fit(X[model_input], y); src, n = "real", len(y)
    else:
        X, y = _synth(mf_data.MODEL_FEATURES, ["RentIncreasePct", "WorkOrderCount", "DaysToLeaseEnd"])
        encoders = {"PropertyName": {}, "PrimaryPetType": {}}
        medians = {c: 0.0 for c in mf_prep.NUMERIC}
        model, auc = _fit(X[model_input], y); n = len(y)
    # Anchor the cohort mean to the v2-validated 32.3% non-renewal rate (logit shift),
    # mirroring recalibrate_h180.py — corrects the selection-bias base rate so the
    # interim model isn't saturated high.
    try:
        tp = model.predict_proba(X[model_input])[:, 1]
        cur = float(np.clip(np.mean(tp), 1e-3, 0.999))
        shift = float(np.log(0.323 / 0.677) - np.log(cur / (1 - cur)))
    except Exception:
        shift = 0.0
    bundle = {"model": model, "feature_columns": mf_data.MODEL_FEATURES,
              "model_input_features": model_input, "encoders": encoders,
              "feature_medians": medians, "bundle_format": "standin",
              "thresholds": {"risk_low_max": 0.25, "risk_medium_max": 0.45, "churn_decision": 0.45},
              "calibration": {"method": "sigmoid+anchor", "prob_floor": 0.01, "prob_ceil": 0.95,
                              "anchor_rate": 0.323, "logit_shift": shift},
              "version": "v0-standin", "trained_at": TS, "algorithm": "lgbm",
              "horizon_days": 180, "label": "ActualChurn", "source": src}
    _save("multifamily_churn", "v0-standin", bundle,
          {"contract": "multifamily-churn", "version": "v0-standin", "feature_count": 45,
           "source": src, "n_train": n, "train_auc": round(auc, 3), "trained_at": TS,
           "note": "INTERIM stand-in. Drop the real churn_model_v4_*.joblib into a new "
                   "version dir and point default.txt to promote it."})


def train_brokerage():
    print("signal-desk-brokerage:")
    df = bk_data.training_data(limit=8000)
    src, n, auc = "synthetic", 0, float("nan")
    if not df.empty and "Label" in df:
        y = pd.to_numeric(df["Label"], errors="coerce")
        keep = y.isin([0, 1]); df, y = df[keep].reset_index(drop=True), y[keep].astype(int).to_numpy()
    else:
        y = np.array([])
    if len(y) >= 150 and len(set(y)) == 2:
        medians = bk_prep.fit_prep(df)
        X = bk_prep.prepare(df, medians)
        model, auc = _fit(X, y); src, n = "real", len(y)
    else:
        X, y = _synth(BK_FEATURES, ["activity_count_90d", "scoop_count_90d", "pipeline_stage"])
        medians = {c: 0.0 for c in BK_FEATURES}
        model, auc = _fit(X, y); n = len(y)
    bundle = {"model": model, "feature_columns": BK_FEATURES, "feature_medians": medians,
              "bundle_format": "standin",
              "thresholds": {"risk_low_max": 0.33, "risk_medium_max": 0.66},
              "calibration": {"method": "sigmoid", "prob_floor": 0.01, "prob_ceil": 0.99},
              "version": "v0-standin", "trained_at": TS, "algorithm": "lgbm",
              "horizon_days": 180, "label": "Label", "source": src}
    _save("signal_desk_brokerage", "v0-standin", bundle,
          {"contract": "signal-desk-brokerage", "version": "v0-standin",
           "feature_count": len(BK_FEATURES), "source": src, "n_train": n,
           "train_auc": round(auc, 3), "trained_at": TS,
           "note": "INTERIM. Real CRE model is data-blocked (few positive labels)."})


if __name__ == "__main__":
    train_multifamily()
    train_brokerage()
    print("DONE")
