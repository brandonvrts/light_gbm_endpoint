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


# ---- Display calibration (serve-layer, rank-preserving) -----------------------
# Domain expectation (confirmed by the model owner): a resident's churn RISK score
# reads in ~10-75%, never 95%. The model's calibrated output has a heavy upper tail
# toward ~1.0 for move-out-notice / expired-lease residents -- those are near-certain,
# already-known departures, not probabilistic risk. The prior hard clip at 0.95 both
# surfaced unrealistic 95%s AND flattened the top of the ranking (dozens of residents
# pinned at the identical ceiling), which defeats "show me the highest-risk residents".
# Instead of clipping, compress the upper tail (KNEE,1] onto (KNEE,CEIL] monotonically:
# scores <= KNEE are shown unchanged; the tail is squeezed under CEIL while STRICTLY
# preserving rank order (the highest raw score still displays highest). Floor at 0.10.
_DISP_FLOOR = 0.10
_DISP_CEIL = 0.75
_DISP_KNEE = 0.60   # below this the calibrated prob is shown unchanged; above, tail compresses


def _display_calibrate(p) -> np.ndarray:
    """Monotone serve-layer squash: model prob -> displayed churn risk in [0.10, 0.75]."""
    p = np.asarray(p, dtype=float).copy()
    hi = p > _DISP_KNEE
    if hi.any():
        p[hi] = _DISP_KNEE + (p[hi] - _DISP_KNEE) * (_DISP_CEIL - _DISP_KNEE) / (1.0 - _DISP_KNEE)
    return np.clip(p, _DISP_FLOOR, _DISP_CEIL)


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
    p = _display_calibrate(p)   # -> displayed churn risk in [0.10, 0.75], rank order preserved
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


def score_lease_window(start: str, end: str, property_ids: list[int] | None = None,
                       as_of: str | None = None, limit: int | None = None,
                       use_cache: bool = True) -> dict:
    """Score the residents whose lease expires within [start, end] -- a lease-expiration
    snapshot. `as_of` (the scoring/eligibility date) defaults to min(today, start): a
    FUTURE window scores with today's knowledge (upcoming expirations); a PAST window is
    point-in-time at the window start (move-outs aren't wrongly filtered and the feature
    SP returns realized ActualChurn for self-verify)."""
    start = data._safe_asof(start)
    end = data._safe_asof(end)
    if end < start:
        start, end = end, start
    asof = data._safe_asof(as_of) if as_of else min(date.today().isoformat(), start)
    pid = list(property_ids) if property_ids else [62, 72, 82, 26]
    key = ("lease-window", tuple(sorted(int(p) for p in pid)), start, end, asof, limit)

    def _fresh():
        hit = _CACHE.get(key)
        return hit[1] if (hit and (time.time() - hit[0]) < _TTL) else None

    if use_cache:
        cached = _fresh()
        if cached is not None:
            return cached
    with _key_lock(key):
        if use_cache:
            cached = _fresh()
            if cached is not None:
                return cached
        scodes = data.scodes_by_lease_window(start, end, pid, asof)
        eligible = len(scodes)
        if limit:
            scodes = scodes[:int(limit)]
        res = score_scodes(scodes, asof)
        res["view"] = "lease-window"
        res["window_start"] = start
        res["window_end"] = end
        res["property_ids"] = pid
        res["eligible_units"] = eligible
        res["scored_units"] = len(res["results"])
        res["cached_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        target = min(int(limit), eligible) if limit else eligible
        if eligible == 0 or res["scored_units"] >= 0.9 * target:
            _CACHE[key] = (time.time(), res)
        return res


def _tenant_payload(r: dict, res: dict) -> dict:
    """Enrich one scored row for a single-tenant lookup: adds stay_probability and a
    natural-language `summary` a chatbot can read back verbatim."""
    p = float(r.get("churn_probability") or 0.0)
    stay = round(1.0 - p, 4)
    name = r.get("full_name") or r.get("scode")
    unit, prop, tier = r.get("unit"), r.get("property"), r.get("risk_tier")
    d2e = r.get("days_to_lease_end")
    notice = bool(r.get("has_notice"))
    if isinstance(d2e, (int, float)):
        lease_txt = (f"their lease ends in {int(d2e)} days" if d2e >= 0
                     else f"their lease ended {abs(int(d2e))} days ago")
    else:
        lease_txt = "their lease-end date is unavailable"
    notice_txt = "a move-out notice is on file" if notice else "no move-out notice is on file"
    summary = (f"{name} (unit {unit}, {prop}) has an estimated {round(p * 100)}% move-out "
               f"risk ({tier}) as of {res.get('as_of')} -- about a {round(stay * 100)}% chance "
               f"of staying. {lease_txt.capitalize()}; {notice_txt}.")
    return {"contract": "multifamily-churn", "model_version": res.get("model_version"),
            "model_source": res.get("model_source"), "as_of": res.get("as_of"),
            "found": True, "scode": r.get("scode"), "full_name": name,
            "unit": unit, "property": prop, "risk_tier": tier,
            "churn_probability": r.get("churn_probability"), "stay_probability": stay,
            "has_notice": notice, "days_to_lease_end": d2e,
            "actual_churn": r.get("actual_churn"),
            "tier_thresholds": res.get("tier_thresholds"), "summary": summary}


def score_one(scode: str, as_of: str | None = None) -> dict:
    """Score a SINGLE resident by SCODE -> churn/stay probability + chatbot-ready summary."""
    sc = (scode or "").strip()
    res = score_scodes([sc], as_of)
    rows = res.get("results", [])
    if not rows:
        return {"contract": "multifamily-churn", "as_of": res.get("as_of"),
                "model_version": res.get("model_version"), "found": False, "scode": sc,
                "summary": f"No current resident was found for SCODE '{sc}'."}
    return _tenant_payload(rows[0], res)


def lookup_tenant(q: str, as_of: str | None = None) -> dict:
    """Resolve a resident NAME -> SCODE. If exactly one current resident matches, the
    scored result (+summary) is embedded under `tenant`; otherwise `matches` lists
    candidates for disambiguation."""
    matches = data.search_tenants_by_name(q, as_of=as_of)
    out = {"contract": "multifamily-churn", "query": (q or "").strip(), "count": len(matches)}
    if len(matches) == 1:
        out["tenant"] = score_one(matches[0]["scode"], as_of)
    else:
        out["matches"] = matches
    return out
