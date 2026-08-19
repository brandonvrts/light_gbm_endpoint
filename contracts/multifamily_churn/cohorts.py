"""Named resident cohorts over the churn model (chatbot surface).

A cohort is a predicate over the point-in-time feature frame of the ACTIVE population,
scored with the deployed bundle via the exact serving path. "How will this cohort
perform" = calibrated expected move-out share at each resident's NEXT lease decision
(v6-roadmap-h60 horizon), so cohort aggregates are honest expected counts.

The v1 catalog is data-validated (2026-07-21 analysis, 955 residents): rent-increase
cohorts were tested and EXCLUDED (survivor bias — a 5%+ increase implies an already-
signed renewal, so they read misleadingly LOW risk), and bare "recent work order" was
excluded as non-differentiating (65% of residents had one in 90d, lift 0.98x); the
maintenance signal that works is first-year + heavy volume (new-and-frustrated).
"""
from __future__ import annotations
import threading
import time
from datetime import date

import numpy as np
import pandas as pd

from core import registry
from contracts.multifamily_churn import contract, data, prepare_v4

_TTL = 3600
_CACHE: dict = {}
_LOCK = threading.Lock()


def _num(df: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_numeric(df.get(name), errors="coerce").fillna(0)


# Each predicate takes the semantic-fixed feature frame (plus the real calendar
# `_d2e` = LeaseToDate - as_of column) and returns a boolean mask.
COHORTS: dict[str, dict] = {
    "holdover-mtm": {
        "label": "Holdover / month-to-month",
        "description": "Lease already expired but still in the unit — no renewal signed. "
                       "The standout risk cohort (~3.6x portfolio churn).",
        "definition": "days_to_lease_end < 0",
        "predicate": lambda d: _num(d, "_d2e") < 0,
    },
    "complaints": {
        "label": "Complaints on file",
        "description": "Residents with complaint-category memos (noise, disturbance, violations).",
        "definition": "ComplaintMemos >= 1",
        "predicate": lambda d: _num(d, "ComplaintMemos") >= 1,
    },
    "new-and-frustrated": {
        "label": "New and frustrated",
        "description": "First-year residents with heavy maintenance volume — dissatisfaction "
                       "before loyalty forms.",
        "definition": "TenureMonths < 12 AND WorkOrders_Last365Days >= 6",
        "predicate": lambda d: (_num(d, "TenureMonths") < 12) & (_num(d, "WorkOrders_Last365Days") >= 6),
    },
    "multi-pet": {
        "label": "Multi-pet households",
        "description": "Residents with 2+ pets.",
        "definition": "PetCount >= 2",
        "predicate": lambda d: _num(d, "PetCount") >= 2,
    },
    "payment-friction": {
        "label": "Payment friction",
        "description": "NSF events or payment-category memos (late fees, delinquency, collections).",
        "definition": "NSFCount >= 1 OR PaymentMemos >= 1",
        "predicate": lambda d: (_num(d, "NSFCount") >= 1) | (_num(d, "PaymentMemos") >= 1),
    },
    "first-year": {
        "label": "First-year residents",
        "description": "Tenure under 12 months.",
        "definition": "TenureMonths < 12",
        "predicate": lambda d: _num(d, "TenureMonths") < 12,
    },
    "pending-decision": {
        "label": "Pending decisions (90d)",
        "description": "Lease expires within 90 days and no move-out notice yet — the "
                       "retention-outreach list (below-average risk, highest leverage).",
        "definition": "0 <= days_to_lease_end <= 90 AND HasNotice = 0",
        "predicate": lambda d: _num(d, "_d2e").between(0, 90) & (_num(d, "HasNotice") == 0),
    },
    "loyal-2y": {
        "label": "Loyal residents (2+ years)",
        "description": "Tenure 24+ months — the stable base (below-average risk).",
        "definition": "TenureMonths >= 24",
        "predicate": lambda d: _num(d, "TenureMonths") >= 24,
    },
    "loyal-5y": {
        "label": "Loyal residents (5+ years)",
        "description": "Tenure 60+ months — most-committed residents (lowest risk).",
        "definition": "TenureMonths >= 60",
        "predicate": lambda d: _num(d, "TenureMonths") >= 60,
    },
    "renewal-veterans": {
        "label": "Renewal veterans",
        "description": "Renewed 2+ times (3+ lease decisions) — proven stayers.",
        "definition": "DecisionNumber >= 3",
        "predicate": lambda d: _num(d, "DecisionNumber") >= 3,
    },
}


def _scored_frame(property_ids: list[int] | None, as_of: str | None,
                  version: str | None = None):
    """Active population scored via the exact serving path (bundle + semantic fixes +
    prepare_v4). Returns (frame incl. `p_churn` calibrated pre-clip + `_d2e` real
    calendar days-to-lease-end, bundle). Cached 1h per (properties, as-of, version).

    `version` overrides the registry default bundle. Views whose semantics need the
    180-day renewal forecast (e.g. the persuadable band — the h60 triage ranker
    compresses almost everyone below the band floor; Phase 0 memo §4, 2026-08-18)
    pass an explicit h180 version; None keeps the serving default."""
    pids = sorted(int(p) for p in (property_ids or data.DEFAULT_PROPERTY_IDS))
    asof = data._safe_asof(as_of)
    key = (tuple(pids), asof, version)
    hit = _CACHE.get(key)
    if hit and time.time() - hit[0] < _TTL:
        return hit[1]
    with _LOCK:
        hit = _CACHE.get(key)
        if hit and time.time() - hit[0] < _TTL:
            return hit[1]
        b = contract._bundle() if version is None else registry.load("multifamily_churn", version)
        scodes = data.active_rentable_scodes(pids, asof)
        df = contract._extract_features(scodes, asof, b)
        if not df.empty:
            if contract._corrected(b):
                df = data.apply_semantic_fixes(df, asof)
            X = prepare_v4.prepare(df, b)
            p = b["model"].predict_proba(b["scaler"].transform(X))[:, 1]
            df = df.reset_index(drop=True)
            df["p_churn"] = np.asarray(p, dtype=float)
            # real calendar distance to lease end (display fix, commit 1198b85) — the
            # holdover / pending predicates key off this, not the SP feature column
            asof_d = date.fromisoformat(asof)
            lease_to = pd.to_datetime(df.get("LeaseToDate"), errors="coerce")
            d2e = (lease_to - pd.Timestamp(asof_d)).dt.days
            df["_d2e"] = d2e.fillna(pd.to_numeric(df.get("DaysToLeaseEnd"), errors="coerce"))
        _CACHE[key] = (time.time(), (df, b))
        return df, b


def _stats(g: pd.DataFrame, base: float, total: int, mid: float) -> dict:
    return {
        "n": int(len(g)),
        "share_pct": round(100 * len(g) / total, 1) if total else 0.0,
        "avg_churn": round(float(g["p_churn"].mean()), 3),
        "lift_vs_portfolio": round(float(g["p_churn"].mean() / base), 2) if base else None,
        "expected_moveouts": round(float(g["p_churn"].sum()), 1),
        "high_risk_n": int((g["p_churn"] > mid).sum()),
        "notice_given_n": int((_num(g, "HasNotice") == 1).sum()),
    }


def _meta(b, asof: str, pids: list[int] | None) -> dict:
    pids = sorted(int(p) for p in (pids or data.DEFAULT_PROPERTY_IDS))
    return {"contract": "multifamily-churn", "model_version": b.get("version"),
            "as_of": asof,
            "properties": [{"property_id": p, "name": data.PROPERTIES.get(p)} for p in pids],
            "horizon_note": "probabilities are calibrated move-out risk at each resident's "
                            "NEXT lease decision (60-day horizon)"}


def catalog(property_ids: list[int] | None = None, as_of: str | None = None) -> dict:
    """All cohorts with live sizes + performance — 'how is every cohort doing'."""
    df, b = _scored_frame(property_ids, as_of)
    asof = data._safe_asof(as_of)
    meta = _meta(b, asof, property_ids)
    if df.empty:
        return {**meta, "view": "cohort-catalog", "population": {"n": 0}, "cohorts": []}
    base = float(df["p_churn"].mean())
    _lo, mid = contract._tiers(b)
    rows = []
    for key, c in COHORTS.items():
        g = df[c["predicate"](df)]
        rows.append({"cohort": key, "label": c["label"], "description": c["description"],
                     "definition": c["definition"],
                     **(_stats(g, base, len(df), mid) if len(g) else
                        {"n": 0, "share_pct": 0.0, "avg_churn": None, "lift_vs_portfolio": None,
                         "expected_moveouts": 0.0, "high_risk_n": 0, "notice_given_n": 0})})
    rows.sort(key=lambda r: (r["lift_vs_portfolio"] or 0), reverse=True)
    return {**meta, "view": "cohort-catalog",
            "population": {"n": int(len(df)), "avg_churn": round(base, 3),
                           "expected_moveouts": round(float(df["p_churn"].sum()), 1)},
            "cohorts": rows}


def report(cohort: str, property_ids: list[int] | None = None, as_of: str | None = None,
           top: int = 10) -> dict:
    """One cohort's predicted performance + its top at-risk members + a chatbot-ready
    natural-language summary."""
    key = (cohort or "").strip().lower()
    if key not in COHORTS:
        raise ValueError(f"unknown cohort '{cohort}' — one of: {', '.join(COHORTS)}")
    c = COHORTS[key]
    df, b = _scored_frame(property_ids, as_of)
    asof = data._safe_asof(as_of)
    meta = _meta(b, asof, property_ids)
    out = {**meta, "view": "cohort-report",
           "cohort": {"key": key, "label": c["label"], "description": c["description"],
                      "definition": c["definition"]}}
    if df.empty:
        return {**out, "population": {"n": 0}, "n": 0, "members_top": [],
                "summary": f"No active residents found, so the {c['label']} cohort is empty."}
    base = float(df["p_churn"].mean())
    lo, mid = contract._tiers(b)
    cal = b.get("calibration") or {}
    g = df[c["predicate"](df)]
    pop = {"n": int(len(df)), "avg_churn": round(base, 3),
           "expected_moveouts": round(float(df["p_churn"].sum()), 1)}
    if g.empty:
        return {**out, "population": pop, "n": 0, "members_top": [],
                "summary": f"No residents currently match the {c['label']} cohort "
                           f"({c['definition']})."}
    st = _stats(g, base, len(df), mid)
    ranked = g.sort_values("p_churn", ascending=False)
    members = []
    for r in ranked.head(max(int(top), 0)).to_dict("records"):
        # member display probability matches the rest of the API: clipped to the
        # bundle's floor/ceil (aggregates above use the unclipped calibrated p)
        p_disp = float(np.clip(r["p_churn"], cal.get("prob_floor", 0.005),
                               cal.get("prob_ceil", 0.95)))
        d2e = r.get("_d2e")
        members.append({
            "scode": str(r.get("SCODE", "")).strip(), "full_name": r.get("FullName"),
            "unit": (str(r.get("UnitCode")).strip() if r.get("UnitCode") is not None else None),
            "property": data.PROPERTIES.get(int(r["PropertyId"])) if pd.notna(r.get("PropertyId")) else r.get("PropertyName"),
            "lease_to": str(r.get("LeaseToDate") or "")[:10] or None,
            "days_to_lease_end": (int(d2e) if pd.notna(d2e) else None),
            "has_notice": int(_i(r.get("HasNotice"))),
            "churn_probability": round(p_disp, 4),
            "risk_tier": contract._tier(p_disp, lo, mid)})
    tiers = {"low": 0, "medium": 0, "high": 0}
    for p in g["p_churn"]:
        tiers[contract._tier(float(np.clip(p, cal.get("prob_floor", 0.005),
                                           cal.get("prob_ceil", 0.95))), lo, mid)] += 1
    summary = (
        f"The {c['label']} cohort ({c['definition']}) has {st['n']} residents "
        f"({st['share_pct']}% of the {pop['n']} scored). Expected performance at their next "
        f"lease decision: {st['avg_churn'] * 100:.1f}% average move-out risk "
        f"({st['lift_vs_portfolio']}x the portfolio average of {base * 100:.1f}%), about "
        f"{st['expected_moveouts']:.0f} expected move-out(s). "
        f"{st['high_risk_n']} are high risk; {st['notice_given_n']} already gave notice.")
    return {**out, "population": pop, **st, "tier_mix": tiers,
            "members_top": members, "summary": summary}


def _i(x) -> int:
    try:
        return int(x) if pd.notna(x) else 0
    except (TypeError, ValueError):
        return 0
