"""Multifamily scoring: SCODEs or buildings -> live features -> churn prob + risk tier.
Wraps the frozen bundle; identical prep at train and inference."""
from __future__ import annotations
from datetime import date, timedelta
from pathlib import Path

import hashlib
import json
import numpy as np
import pandas as pd

from core import registry
from contracts.multifamily_churn import data, explain, prepare_features as prep

import time
import threading
_CACHE: dict = {}
_LOCKS: dict = {}                 # per-(key) compute locks: dedupe identical views,
_LOCKS_GUARD = threading.Lock()   # but let different views / backtests run concurrently
_TTL = 3600  # the SP recompute is heavy; serve cached building scores for 1h

_ROOT = Path(__file__).resolve().parent.parent.parent
_CACHE_DIR = _ROOT / "monitoring" / "report_cache"   # Phase 0.3: survive restarts
_DRIFT_LOG = _ROOT / "monitoring" / "calibration.log"  # Phase 1.5


def _disk_path(key) -> Path:
    h = hashlib.sha1(json.dumps(key, default=str, sort_keys=True).encode()).hexdigest()
    return _CACHE_DIR / f"{h}.json"


def _cache_get(key):
    """Memory cache, then disk (Phase 0.3): a restart keeps the warm months warm."""
    hit = _CACHE.get(key)
    if hit and (time.time() - hit[0]) < _TTL:
        return hit[1]
    p = _disk_path(key)
    if p.exists():
        try:
            rec = json.loads(p.read_text())
            if (time.time() - rec["ts"]) < _TTL:
                _CACHE[key] = (rec["ts"], rec["payload"])
                return rec["payload"]
        except Exception:
            pass
    return None


def _cache_put(key, payload):
    _CACHE[key] = (time.time(), payload)
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _disk_path(key).write_text(json.dumps({"ts": time.time(), "payload": payload},
                                              default=str))
    except Exception:      # cache persistence must never break scoring
        pass


def _store_cache_disk(key, payload):
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _disk_path(key).write_text(json.dumps({"ts": time.time(), "payload": payload},
                                              default=str))
    except Exception:
        pass


def _log_calibration_drift(report: dict) -> None:
    """Phase 1.5: when a past month is settled (few pending), append predicted-vs-realized
    to monitoring/calibration.log (one line per month+model). Feeds the intercept refit and
    surfaces drift instead of discovering it a year later. Never raises."""
    try:
        end = report.get("window_end") or ""
        if end >= date.today().isoformat():
            return                                   # month not over
        n = report.get("cohort_size") or 0
        pend = (report.get("outcomes") or {}).get("pending", 0)
        if not n or pend > 0.15 * n:
            return                                   # not settled enough to judge
        rates = report.get("rates") or {}
        pred = rates.get("predicted_churn_rate")
        real = rates.get("realized_churn_rate")
        if pred is None or real is None:
            return
        mv = report.get("model_version")
        month = report.get("month")
        _DRIFT_LOG.parent.mkdir(parents=True, exist_ok=True)
        if _DRIFT_LOG.exists():
            for line in _DRIFT_LOG.read_text().splitlines():
                try:
                    r = json.loads(line)
                    if r.get("month") == month and r.get("model_version") == mv:
                        return                       # already logged this month+model
                except Exception:
                    continue
        rec = {"logged_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "month": month,
               "model_version": mv, "cohort_size": n, "pending": pend,
               "predicted_churn_rate": pred, "realized_churn_rate": real,
               "gap": round(pred - real, 4),
               "alert": abs(pred - real) > 0.05}
        with _DRIFT_LOG.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
    except Exception:
        pass


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


def _corrected(bundle: dict) -> bool:
    """v5/v6 bundles train AND serve on corrected feature semantics (person-level tenure,
    calendar days-to-lease-end) via data.apply_semantic_fixes and are served without the
    legacy display squash."""
    return str(bundle.get("feature_semantics", "")).startswith(("v5", "v6", "v7"))


def _extract_features(scodes: list[str], as_of: str | None, bundle: dict) -> pd.DataFrame:
    """Pick the feature source for the bundle: v6 uses the set-based, as-of-anchored
    point-in-time query (Phase 2.1 -- one query per cohort, no lease-end feature skew, no
    past-month outcome leakage); v5/legacy use the per-resident SP."""
    if str(bundle.get("feature_semantics", "")).startswith(("v6", "v7")):
        return data.cohort_features_asof(scodes, as_of)
    return data.features_by_scodes(scodes, as_of)


def _predict(df: pd.DataFrame, bundle: dict, as_of: str | None = None,
             include_model_prob: bool = False) -> list[dict]:
    cal = bundle.get("calibration") or {}
    corrected = _corrected(bundle)
    Xs = expl_contrib = None
    if bundle.get("bundle_format") == "standin":
        X = prep.prepare(df, bundle["encoders"], bundle["feature_medians"])
        feats = bundle.get("model_input_features", bundle["feature_columns"])
        p = bundle["model"].predict_proba(X[feats])[:, 1]
        shift = cal.get("logit_shift")
        if shift:
            p = np.clip(p, 1e-4, 1 - 1e-4)
            p = 1.0 / (1.0 + np.exp(-(np.log(p / (1 - p)) + shift)))
    else:  # real production bundle (scaler + LabelEncoders + calibrator)
        from contracts.multifamily_churn import prepare_v4
        if corrected:
            df = data.apply_semantic_fixes(df, as_of)
        X = prepare_v4.prepare(df, bundle)
        Xs = bundle["scaler"].transform(X)
        p = bundle["model"].predict_proba(Xs)[:, 1]
        expl_contrib = explain.contributions(bundle, Xs)
    p_model = np.asarray(p, dtype=float).copy()   # calibrated model prob, pre display transform
    if corrected:
        # v5/v6 probabilities are smooth and cohort-honest -- serve as-is (no snapping/
        # squash), clipped to the bundle's floor/ceil.
        p = np.clip(np.asarray(p, dtype=float),
                    cal.get("prob_floor", 0.005), cal.get("prob_ceil", 0.95))
    else:
        p = _display_calibrate(p)   # legacy display squash for pre-v5 bundles
    lo, mid = _tiers(bundle)
    rows = df.reset_index(drop=True)
    try:
        asof_d = date.fromisoformat(str(as_of)[:10]) if as_of else date.today()
    except ValueError:
        asof_d = date.today()
    out = []
    for i in range(len(rows)):
        r = rows.iloc[i]
        unit = r.get("UnitCode")
        # The SP's DaysToLeaseEnd feature is anchored to the renewal DECISION date (0 by
        # construction for many rows) -- display the real calendar distance from as_of.
        lease_to = str(r.get("LeaseToDate") or "")[:10]
        try:
            d2e = (date.fromisoformat(lease_to) - asof_d).days
        except ValueError:
            lease_to, d2e = None, r.get("DaysToLeaseEnd")
        prop = r.get("PropertyName")
        if prop in (None, "", "Unknown"):   # SP only names the 4 trained properties
            try:
                prop = data.PROPERTIES.get(int(r.get("PropertyId")), prop)
            except (TypeError, ValueError):
                pass
        rent = r.get("CurrentRent")
        if rent is None or (isinstance(rent, float) and pd.isna(rent)) or float(rent or 0) <= 0:
            rent = r.get("ActiveRent")
        try:
            rent = round(float(rent), 2) if rent is not None and float(rent) > 0 else None
        except (TypeError, ValueError):
            rent = None
        row = {
            "scode": str(r.get("SCODE", "")).strip(),
            "full_name": r.get("FullName"),
            "unit": str(unit).strip() if unit is not None else None,
            "property": prop,
            "lease_to": lease_to,
            "days_to_lease_end": d2e,
            "tenure_months": (round(float(r["TenureMonths"]), 1)
                              if pd.notna(r.get("TenureMonths")) else None),
            "monthly_rent": rent,
            "has_notice": r.get("HasNotice"),
            "actual_churn": r.get("ActualChurn"),
            "churn_probability": round(float(p[i]), 4),
            "risk_tier": _tier(float(p[i]), lo, mid),
        }
        if expl_contrib is not None:
            e = explain.why(bundle, X, expl_contrib, i, prop_name=prop)
            row["why"] = e["why"]
            row["drivers"] = e["drivers"]
        if include_model_prob:
            # The UNsquashed calibrated probability. churn_probability is display-
            # calibrated (capped at 0.75) for per-resident readability; aggregating an
            # expected churn COUNT/RATE must use this one or the tail is understated.
            row["model_probability"] = round(float(p_model[i]), 4)
        out.append(row)
    return out


def score_scodes(scodes: list[str], as_of: str | None = None,
                 include_model_prob: bool = False) -> dict:
    as_of = as_of or date.today().isoformat()
    b = _bundle()
    df = _extract_features(scodes, as_of, b)
    source = b.get("source") or ("standin" if b.get("bundle_format") == "standin" else "real-production")
    lo, mid = _tiers(b)
    return {"contract": "multifamily-churn", "model_version": b.get("version"),
            "model_source": source, "as_of": as_of,
            "tier_thresholds": {"low_max": lo, "medium_max": mid},  # single source of truth
            "count": 0 if df.empty else int(len(df)),
            "results": [] if df.empty else _predict(df, b, as_of, include_model_prob)}


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
    pid = list(property_ids) if property_ids else list(data.DEFAULT_PROPERTY_IDS)
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


def _month_bounds(month: str) -> tuple[str, str]:
    """'YYYY-MM' (or any date in the month) -> (first-day, last-day) ISO strings."""
    m = str(month or "").strip()[:7]
    try:
        y, mo = int(m[:4]), int(m[5:7])
        first = date(y, mo, 1)
    except (ValueError, IndexError):
        t = date.today()
        first = date(t.year, t.month, 1)
    nxt = (date(first.year + 1, 1, 1) if first.month == 12
           else date(first.year, first.month + 1, 1))
    return first.isoformat(), (nxt - timedelta(days=1)).isoformat()


def monthly_churn_report(month: str, property_ids: list[int] | None = None,
                         include_scores: bool = True, use_cache: bool = True) -> dict:
    """Month-over-month churn report: everyone whose lease expired (or expires) in the
    given calendar month, their renewal OUTCOME as known today (renewed / churned /
    pending), and -- optionally -- the model's churn score as of the month start, so
    predictions can be compared against what actually happened.

    Cohort comes from Lease_History (point-in-time, survives Yardi's in-place renewal
    updates), NOT the mutable TENANT.DTLEASETO -- see data.lease_window_cohort."""
    start, end = _month_bounds(month)
    asof = min(date.today().isoformat(), start)
    pid = list(property_ids) if property_ids else list(data.DEFAULT_PROPERTY_IDS)
    key = ("churn-report", _bundle().get("version"),
           tuple(sorted(int(p) for p in pid)), start, include_scores)

    if use_cache:
        cached = _cache_get(key)
        if cached is not None:
            return cached
    with _key_lock(key):
        if use_cache:
            cached = _cache_get(key)
            if cached is not None:
                return cached
        cohort = data.lease_window_cohort(start, end, pid, asof)
        rows: list[dict] = []
        outcomes: dict[int, str] = {}
        if not cohort.empty:
            outcomes = data.renewal_outcomes(cohort["HMYPERSON"].tolist(), start, end)
        scored: dict[str, dict] = {}
        b = _bundle()
        res_meta: dict = {}
        if include_scores and not cohort.empty:
            res_meta = score_scodes(cohort["SCODE"].tolist(), asof, include_model_prob=True)
            scored = {r["scode"].lower(): r for r in res_meta.get("results", [])}
        for _, c in cohort.iterrows():
            sc = str(c["SCODE"]).strip()
            r = dict(scored.get(sc.lower()) or {"scode": sc})
            r.setdefault("property", data.PROPERTIES.get(int(c["PropertyId"])))
            r["lease_to"] = str(c["LeaseTo"])[:10]
            r["outcome"] = outcomes.get(int(c["HMYPERSON"]), "pending")
            rows.append(r)
        # Rank by the UNsquashed model prob where available -- the display prob ties at
        # the 0.75 ceiling for near-certain departures and would shuffle those rows.
        rows.sort(key=lambda r: (-(r.get("model_probability")
                                   or r.get("churn_probability") or 0), r["scode"]))
        n = len(rows)
        # Phase 1.1: `transferred` (internal unit move -> resident retained) is a fourth
        # outcome, counted as retention, NOT churn.
        counts = {o: sum(1 for r in rows if r["outcome"] == o)
                  for o in ("renewed", "churned", "pending", "transferred")}
        decided = counts["renewed"] + counts["churned"] + counts["transferred"]
        retained = counts["renewed"] + counts["transferred"]
        # Model hit-rate on decided churners: those the model had flagged medium/high.
        flagged = [r for r in rows if r["outcome"] == "churned"
                   and r.get("risk_tier") in ("medium", "high")]
        # Predicted churn from UNsquashed calibrated probs (sums to an expected count).
        predicted = None
        rev = None
        if include_scores:
            probs = [r["model_probability"] for r in rows
                     if r.get("model_probability") is not None]
            pend_probs = [r["model_probability"] for r in rows
                          if r["outcome"] == "pending"
                          and r.get("model_probability") is not None]
            if probs:
                exp_churns = float(sum(probs))
                proj_churns = counts["churned"] + float(sum(pend_probs))
                predicted = {
                    "expected_churns": round(exp_churns, 1),
                    "predicted_churn_rate": round(exp_churns / len(probs), 4),
                    "projected_churns": round(proj_churns, 1),
                    "projected_churn_rate": round(proj_churns / n, 4) if n else None,
                    "basis": ("predicted_* = model only (mean calibrated prob, whole "
                              "cohort); projected_* = decided outcomes as fact + model "
                              "probs for pending residents"),
                }
            # Phase 3.1: expected monthly revenue at risk = sum(churn prob x rent).
            rented = [(r.get("model_probability"), r.get("monthly_rent")) for r in rows
                      if r.get("monthly_rent")]
            if rented:
                exp_rev = sum((mp or 0) * rent for mp, rent in rented)
                churned_rent = sum(r.get("monthly_rent") or 0 for r in rows
                                   if r["outcome"] == "churned")
                pend_rev = sum((r.get("model_probability") or 0) * (r.get("monthly_rent") or 0)
                               for r in rows if r["outcome"] == "pending")
                rev = {"expected_monthly_revenue_at_risk": round(exp_rev, 0),
                       "projected_monthly_revenue_at_risk": round(churned_rent + pend_rev, 0),
                       "cohort_monthly_rent": round(sum(rt for _, rt in rented), 0),
                       "residents_with_rent": len(rented), "currency": "USD"}
        lo, mid = _tiers(b)
        # Phase 3.2: one labeled rates block + a single canonical headline field.
        churn_rate = round(counts["churned"] / decided, 4) if decided else None
        proj_rate = (predicted or {}).get("projected_churn_rate")
        rates = {
            "realized_churn_rate": churn_rate,
            "realized_retention_rate": round(retained / decided, 4) if decided else None,
            "predicted_churn_rate": (predicted or {}).get("predicted_churn_rate"),
            "projected_churn_rate": proj_rate,
            "headline_churn_rate": proj_rate if proj_rate is not None else churn_rate,
            "headline_basis": ("projected (decided outcomes + model for pending)"
                               if proj_rate is not None else
                               "realized (decided outcomes only; churned / decided)"),
            "decided_denominator": decided,
            "definitions": {
                "realized_churn_rate": "churned / (renewed + churned + transferred)",
                "projected_churn_rate": "(churned + sum model prob over pending) / cohort_size",
                "note": "transferred = internal unit move (retained, not churn)",
            },
        }
        out = {"contract": "multifamily-churn", "view": "monthly-churn-report",
               "month": start[:7], "window_start": start, "window_end": end,
               "as_of": asof, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "model_version": res_meta.get("model_version") or b.get("version"),
               "property_ids": pid,
               "tier_thresholds": {"low_max": lo, "medium_max": mid},
               "tier_labels": b.get("tier_labels"),
               "cohort_size": n,
               "outcomes": counts,
               "rates": rates,
               # back-compat scalars (headline is canonical; see `rates`)
               "churn_rate": churn_rate,
               "renewal_rate": round(retained / decided, 4) if decided else None,
               "model_flagged_churners": (round(len(flagged) / counts["churned"], 4)
                                          if counts["churned"] and include_scores else None),
               "predicted": predicted,
               "revenue_at_risk": rev,
               "results": rows}
        if n:
            _cache_put(key, out)
        if include_scores:
            _log_calibration_drift(out)   # Phase 1.5 (no-op unless the month is decided)
        return out


def _report_summary(rep: dict) -> str:
    """One-paragraph natural-language readback of a monthly churn report -- written so a
    chatbot can quote it verbatim (mirrors score_one's `summary`). Always drawn from the
    FULL report, regardless of any top/outcome trimming of `results`."""
    label = date.fromisoformat(rep["window_start"]).strftime("%B %Y")
    n, c = rep["cohort_size"], rep["outcomes"]
    xfer = f" {c['transferred']} transferred units (retained)," if c.get("transferred") else ""
    parts = [f"Of the {n} residents whose leases expire in {label}, "
             f"{c['renewed']} have renewed,{xfer} {c['churned']} have a move-out or notice "
             f"on file, and {c['pending']} are still undecided."]
    hr = (rep.get("rates") or {}).get("headline_churn_rate")
    if hr is not None:
        parts.append(f"Projected churn rate: {round(hr * 100)}%.")
    rev = rep.get("revenue_at_risk") or {}
    if rev.get("projected_monthly_revenue_at_risk"):
        parts.append(f"Estimated monthly revenue at risk: "
                     f"${int(rev['projected_monthly_revenue_at_risk']):,}.")
    risks = [r for r in rep["results"] if r.get("outcome") == "pending"
             and r.get("churn_probability") is not None][:3]
    if risks:
        txt = "; ".join(f"{r.get('full_name') or r['scode']} (unit {r.get('unit')}, "
                        f"{r.get('property')}) at {round(r['churn_probability'] * 100)}%"
                        for r in risks)
        parts.append(f"Highest-risk undecided residents: {txt}.")
    return " ".join(parts)


def churn_report_view(month: str, property_ids: list[int] | None = None,
                      include_scores: bool = True, top: int | None = None,
                      outcome: str | None = None) -> dict:
    """Serve-time view over monthly_churn_report for chatbots/UIs: optional outcome
    filter (renewed|churned|pending) + top-N truncation (rows are already ranked by
    churn probability, highest first) + a natural-language `summary`. The full report
    stays cached; this trims a COPY, never the cached dict."""
    rep = monthly_churn_report(month, property_ids, include_scores)
    rows = rep["results"]
    if outcome:
        rows = [r for r in rows if r.get("outcome") == str(outcome).strip().lower()]
    total = len(rows)
    if top:
        rows = rows[:max(1, int(top))]
    out = dict(rep)
    out["results"] = rows
    out["results_total"] = total
    out["results_shown"] = len(rows)
    if outcome:
        out["filter_outcome"] = str(outcome).strip().lower()
    out["summary"] = _report_summary(rep)
    return out


def churn_trend(from_month: str, to_month: str, property_ids: list[int] | None = None,
                include_predicted: bool = False) -> dict:
    """Month-over-month summary series: cohort size, renewed/churned/pending and churn
    rate for each calendar month in [from, to]. Default is outcome-only (pure SQL, fast).
    include_predicted=True adds the model's predicted/projected churn rate per month --
    that scores every cohort (slow on first call, ~1 min/uncached month; results cache
    for 1h), so the span is capped at 12 months in that mode (else 36)."""
    s, _ = _month_bounds(from_month)
    e, _ = _month_bounds(to_month)
    if e < s:
        s, e = e, s
    pid = list(property_ids) if property_ids else list(data.DEFAULT_PROPERTY_IDS)
    months: list[str] = []
    cur = date.fromisoformat(s)
    stop = date.fromisoformat(e)
    cap = 12 if include_predicted else 36
    while cur <= stop and len(months) < cap:
        months.append(cur.isoformat()[:7])
        cur = (date(cur.year + 1, 1, 1) if cur.month == 12
               else date(cur.year, cur.month + 1, 1))
    series = []
    for m in months:
        rep = monthly_churn_report(m, pid, include_scores=include_predicted)
        row = {k: rep[k] for k in ("month", "cohort_size", "outcomes",
                                   "churn_rate", "renewal_rate")}
        row["headline_churn_rate"] = (rep.get("rates") or {}).get("headline_churn_rate")
        if include_predicted:
            pr = rep.get("predicted") or {}
            row["predicted_churn_rate"] = pr.get("predicted_churn_rate")
            row["projected_churn_rate"] = pr.get("projected_churn_rate")
            rev = rep.get("revenue_at_risk") or {}
            row["projected_monthly_revenue_at_risk"] = rev.get("projected_monthly_revenue_at_risk")
        series.append(row)
    return {"contract": "multifamily-churn", "view": "churn-trend",
            "from": months[0] if months else None, "to": months[-1] if months else None,
            "property_ids": pid, "include_predicted": include_predicted,
            "months": series}


def _tenant_payload(r: dict, res: dict) -> dict:
    """Enrich one scored row for a single-tenant lookup: adds stay_probability and a
    natural-language `summary` a chatbot can read back verbatim."""
    p = float(r.get("churn_probability") or 0.0)
    stay = round(1.0 - p, 4)
    name = r.get("full_name") or r.get("scode")
    unit, prop, tier = r.get("unit"), r.get("property"), r.get("risk_tier")
    d2e = r.get("days_to_lease_end")
    notice = bool(r.get("has_notice"))
    lt = r.get("lease_to")
    on = f" ({lt})" if lt else ""
    if isinstance(d2e, (int, float)):
        lease_txt = (f"their lease ends in {int(d2e)} days{on}" if d2e >= 0
                     else f"their lease ended {abs(int(d2e))} days ago{on}")
    else:
        lease_txt = "their lease-end date is unavailable"
    notice_txt = "a move-out notice is on file" if notice else "no move-out notice is on file"
    summary = (f"{name} (unit {unit}, {prop}) has an estimated {round(p * 100)}% move-out "
               f"risk ({tier}) as of {res.get('as_of')} -- about a {round(stay * 100)}% chance "
               f"of staying. {lease_txt.capitalize()}; {notice_txt}.")
    if r.get("why"):
        summary += " " + r["why"]
    return {"contract": "multifamily-churn", "model_version": res.get("model_version"),
            "model_source": res.get("model_source"), "as_of": res.get("as_of"),
            "found": True, "scode": r.get("scode"), "full_name": name,
            "unit": unit, "property": prop, "risk_tier": tier,
            "churn_probability": r.get("churn_probability"), "stay_probability": stay,
            "has_notice": notice, "lease_to": r.get("lease_to"), "days_to_lease_end": d2e,
            "tenure_months": r.get("tenure_months"),
            "actual_churn": r.get("actual_churn"),
            "why": r.get("why"), "drivers": r.get("drivers"),
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
