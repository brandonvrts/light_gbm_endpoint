"""Outreach tracking (Phase 3.3): record which at-risk residents the property team
contacted and what happened, so retention outreach can later be measured against churn
outcomes (uplift), not just ranked.

The SSMS source is READ-ONLY (core.db never writes), so this is a local append-only
JSONL store under monitoring/. Each record snapshots the resident's churn score AT
contact time -- that snapshot is what makes a future uplift analysis honest (compare
contacted vs comparable non-contacted residents at the same predicted risk).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

_STORE = Path(__file__).resolve().parent.parent.parent / "monitoring" / "outreach.jsonl"
# incentive_* actions are the persuadables measurement loop: OFFERS (accepted AND
# declined) are what TRANS can never show — logging them creates the denominator a real
# flip rate needs. Log incentive_offered at offer time, then incentive_accepted /
# incentive_declined when the resident responds.
_ACTIONS = {"called", "emailed", "visited", "offer_sent", "renewed_offer",
            "no_answer", "note",
            "incentive_offered", "incentive_accepted", "incentive_declined"}

# Typed offers (Phase 0 measurement discipline, 2026-08-18): a flip rate per OFFER TYPE
# is only measurable if the type is logged at offer time. `other` exists so a novel
# offer never blocks logging — but it can't be analyzed, so name new types promptly.
OFFER_TYPES = {"free_months_1", "free_months_2", "free_months_3",
               "free_garage_new", "free_garage_waiver", "rent_discount_pct",
               "post_notice_rescue_1mo", "other"}
# How the dollars land — a discount baked into base compounds and degrades face rent;
# a one-time credit does not (Phase 0 §12.1). Cost models need this, not just an amount.
OFFER_STRUCTURES = {"one_time_credit", "recurring_discount_term",
                    "recurring_discount_base", "waiver"}


def record(scode: str, action: str, staff: str | None = None, note: str | None = None,
           month: str | None = None, churn_probability: float | None = None,
           risk_tier: str | None = None, incentive_cost: float | None = None,
           offer_type: str | None = None, offer_structure: str | None = None,
           offer_value: float | None = None, property_id: int | None = None,
           bedroom: int | None = None, arm: str | None = None,
           posting_account: int | None = None) -> dict:
    """Append one outreach event. `action` is free-ish but validated against a known set
    (falls back to 'note'). The churn_probability/risk_tier snapshot should be the values
    shown to staff at contact time (pass them through from the report row).

    incentive_* actions REQUIRE `offer_type` (one of OFFER_TYPES) — untyped offers can
    never yield a per-offer flip rate, which is the entire point of this log. Optional
    offer fields: `offer_structure` (OFFER_STRUCTURES), `offer_value` (months / pct / $
    per the type), `incentive_cost` (expected $ if consumed), `bedroom`, `property_id`,
    `arm` ('treated'/'control' for randomized trials), `posting_account` (TRANS
    HRETENTIONACCT the credit will post to — keeps the consumed-side ledger joinable)."""
    sc = (scode or "").strip()
    if not sc:
        raise ValueError("scode required")
    act = (action or "note").strip().lower()
    if act not in _ACTIONS:
        act = "note"
    ot = (offer_type or "").strip().lower() or None
    if act.startswith("incentive_"):
        if ot is None:
            raise ValueError(
                "offer_type required for incentive_* actions (one of: "
                f"{', '.join(sorted(OFFER_TYPES))}) — untyped offers can't be measured")
        if ot not in OFFER_TYPES:
            raise ValueError(f"unknown offer_type '{ot}' — one of: "
                             f"{', '.join(sorted(OFFER_TYPES))}")
    os_ = (offer_structure or "").strip().lower() or None
    if os_ is not None and os_ not in OFFER_STRUCTURES:
        raise ValueError(f"unknown offer_structure '{os_}' — one of: "
                         f"{', '.join(sorted(OFFER_STRUCTURES))}")
    arm_ = (arm or "").strip().lower() or None
    if arm_ is not None and arm_ not in ("treated", "control"):
        raise ValueError("arm must be 'treated' or 'control'")
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "scode": sc, "action": act,
           "staff": (staff or "").strip() or None, "note": (note or "").strip() or None,
           "month": (month or "").strip() or None,
           "churn_probability": churn_probability, "risk_tier": risk_tier,
           "incentive_cost": incentive_cost,
           "offer_type": ot, "offer_structure": os_, "offer_value": offer_value,
           "property_id": property_id, "bedroom": bedroom, "arm": arm_,
           "posting_account": posting_account}
    _STORE.parent.mkdir(parents=True, exist_ok=True)
    with _STORE.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    return rec


def _read() -> list[dict]:
    if not _STORE.exists():
        return []
    out = []
    for line in _STORE.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def history(scode: str | None = None, month: str | None = None,
            limit: int = 200) -> dict:
    """List outreach events, newest first, optionally filtered by scode and/or month."""
    recs = _read()
    if scode:
        s = scode.strip().lower()
        recs = [r for r in recs if (r.get("scode") or "").lower() == s]
    if month:
        recs = [r for r in recs if (r.get("month") or "").startswith(month.strip()[:7])]
    recs = list(reversed(recs))[:max(1, int(limit))]
    return {"contract": "multifamily-churn", "view": "outreach-history",
            "count": len(recs), "scode": scode, "month": month, "events": recs}


def summary() -> dict:
    """Coverage snapshot: contacted residents by month + action mix. A full uplift
    (contacted vs matched-risk non-contacted -> realized churn) needs outcomes joined
    downstream; this reports what has been logged so far."""
    recs = _read()
    by_month: dict[str, set] = {}
    by_action: dict[str, int] = {}
    for r in recs:
        m = r.get("month") or (r.get("ts") or "")[:7]
        by_month.setdefault(m, set()).add(r.get("scode"))
        by_action[r.get("action")] = by_action.get(r.get("action"), 0) + 1
    return {"contract": "multifamily-churn", "view": "outreach-summary",
            "total_events": len(recs),
            "residents_contacted_by_month": {m: len(s) for m, s in sorted(by_month.items())},
            "actions": by_action}
