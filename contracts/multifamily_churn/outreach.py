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
_ACTIONS = {"called", "emailed", "visited", "offer_sent", "renewed_offer",
            "no_answer", "note"}


def record(scode: str, action: str, staff: str | None = None, note: str | None = None,
           month: str | None = None, churn_probability: float | None = None,
           risk_tier: str | None = None) -> dict:
    """Append one outreach event. `action` is free-ish but validated against a known set
    (falls back to 'note'). The churn_probability/risk_tier snapshot should be the values
    shown to staff at contact time (pass them through from the report row)."""
    sc = (scode or "").strip()
    if not sc:
        raise ValueError("scode required")
    act = (action or "note").strip().lower()
    if act not in _ACTIONS:
        act = "note"
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "scode": sc, "action": act,
           "staff": (staff or "").strip() or None, "note": (note or "").strip() or None,
           "month": (month or "").strip() or None,
           "churn_probability": churn_probability, "risk_tier": risk_tier}
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
