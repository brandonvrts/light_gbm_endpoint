"""Multifamily scoring: SCODEs or buildings -> live features -> churn prob + risk tier.
Wraps the frozen bundle; identical prep at train and inference."""
from __future__ import annotations
from datetime import date

import numpy as np
import pandas as pd

from core import registry
from contracts.multifamily_churn import data, prepare_features as prep

import time
import threading
_CACHE: dict = {}
_LOCKS: dict = {}                 # per-(key) compute locks: dedupe identical views,
_LOCKS_GUARD = threading.Lock()   # but let different views / backtests run concurrently
_TTL = 3600  # the SP recompute is heavy; serve cached building scores for 1h


def _key_lock(key) -> threading.Lock:
    with _LOCKS_GUARD:
        lk = _LOCKS.get(key)
        if lk is None:
            lk = _LOCKS[key] = threading.Lock()
        return lk


def _bundle() -> dict:
    return registry.load("multifamily_churn")


def _tiers(bundle: dict) -> tuple[float, float]:
    """Risk-tier cutpoints, guarded: some bundles ship degenerate/inverted values."""
    th = bundle.get("thresholds") or {}
    lo, mid = th.get("risk_low_max"), th.get("risk_medium_max")
    if not (isinstance(lo, (int, float)) and isinstance(mid, (int, float)) and 0 < lo < mid < 1):
        lo, mid = 0.25, 0.45
    return lo, mid


def _tier(p: float, lo: float, mid: float) -> str:
    return "low" if p < lo else ("medium" if p < mid else "high")


def _predict(df: pd.DataFrame, bundle: dict) -> list[dict]:
    cal = bundle.get("calibration") or {}
    if bundle.get("bundle_format") == "standin":
        X = prep.prepare(df, bundle["encoders"], bundle["feature_medians"])
        feats = bundle.get("model_input_features", bundle["feature_columns"])
        p = bundle["model"].predict_proba(X[feats])[:, 1]
        shift = cal.get("logit_shift")
        if shift:
            p = np.clip(p, 1e-4, 1 - 1e-4)
            p = 1.0 / (1.0 + np.exp(-(np.log(p / (1 - p)) + shift)))
    else:  # real production v4 bundle (scaler + LabelEncoders + PlattRescaler)
        from contracts.multifamily_churn import prepare_v4
        X = prepare_v4.prepare(df, bundle)
        p = bundle["model"].predict_proba(bundle["scaler"].transform(X))[:, 1]
    p = np.clip(p, cal.get("prob_floor", 0.01), cal.get("prob_ceil", 0.99))
    lo, mid = _tiers(bundle)
    rows = df.reset_index(drop=True)
    out = []
    for i in range(len(rows)):
        r = rows.iloc[i]
        out.append({
            "scode": str(r.get("SCODE", "")).strip(),
            "full_name": r.get("FullName"),
            "unit": r.get("UnitCode"),
            "property": r.get("PropertyName"),
            "days_to_lease_end": r.get("DaysToLeaseEnd"),
            "has_notice": r.get("HasNotice"),
            "actual_churn": r.get("ActualChurn"),
            "churn_probability": round(float(p[i]), 4),
            "risk_tier": _tier(float(p[i]), lo, mid),
        })
    return out


def score_scodes(scodes: list[str], as_of: str | None = None) -> dict:
    as_of = as_of or date.today().isoformat()
    b = _bundle()
    df = data.features_by_scodes(scodes, as_of)
    source = b.get("source") or ("standin" if b.get("bundle_format") == "standin" else "real-production")
    lo, mid = _tiers(b)
    return {"contract": "multifamily-churn", "model_version": b.get("version"),
            "model_source": source, "as_of": as_of,
            "tier_thresholds": {"low_max": lo, "medium_max": mid},  # single source of truth
            "count": 0 if df.empty else int(len(df)),
            "results": [] if df.empty else _predict(df, b)}


def score_buildings(property_ids: list[int], as_of: str | None = None,
                    limit: int | None = None, use_cache: bool = True) -> dict:
    key = (tuple(sorted(int(p) for p in property_ids)), data._safe_asof(as_of), limit)

    def _fresh():
        hit = _CACHE.get(key)
        return hit[1] if (hit and (time.time() - hit[0]) < _TTL) else None

    if use_cache:
        cached = _fresh()
        if cached is not None:
            return cached
    # Dedupe concurrent callers of the SAME view (they wait, then hit the cache) while
    # letting DIFFERENT views (e.g. the warm's full portfolio + a historical backtest)
    # compute in parallel instead of one blocking the other.
    with _key_lock(key):
        if use_cache:
            cached = _fresh()
            if cached is not None:
                return cached
        scodes = data.active_rentable_scodes(property_ids, as_of)  # current resident per rentable unit
        eligible = len(scodes)
        if limit:                  # optional cap; default None = score ALL occupied units
            scodes = scodes[:int(limit)]
        res = score_scodes(scodes, as_of)
        res["property_ids"] = list(property_ids)
        res["eligible_units"] = eligible
        res["scored_units"] = len(res["results"])
        res["cached_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        # only cache a COMPLETE result; never persist a partial (e.g. transient SP hiccup).
        # "complete" respects an explicit limit -> target = min(limit, eligible), so limited/
        # backtest runs cache too (previously they never did -> every re-run recomputed).
        target = min(int(limit), eligible) if limit else eligible
        if eligible == 0 or res["scored_units"] >= 0.9 * target:
            _CACHE[key] = (time.time(), res)
        return res
