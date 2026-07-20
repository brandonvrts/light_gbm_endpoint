"""Brokerage lead-outcome tracking (Phase 3.2): record which scorecard / space-demand
leads brokers actually pursue and how they resolve, so the ranking can eventually be
measured by UPLIFT (did pursuing high-ranked leads convert more?) rather than by ranking
metrics alone.

Read-only SSMS seam -> local append-only JSONL under monitoring/. Each event snapshots the
rank signals shown at contact time (rea_score / win_prob / opportunity_tier) so a later
uplift study can compare pursued vs comparable non-pursued companies at equal predicted
rank."""
from __future__ import annotations

import json
import time
from pathlib import Path

_STORE = Path(__file__).resolve().parent.parent.parent / "monitoring" / "brokerage_leads.jsonl"
_ACTIONS = {"pursued", "called", "emailed", "meeting", "proposal_sent",
            "won", "lost", "disqualified", "skip", "note"}


def record(company_id: int | None = None, hmy: int | None = None, action: str = "note",
           staff: str | None = None, note: str | None = None, market: str | None = None,
           rea_score: float | None = None, win_prob: float | None = None,
           conversion_probability: float | None = None,
           opportunity_tier: str | None = None, source: str | None = None) -> dict:
    """Append one lead-outcome event. Identify by company_id (ZI) and/or hMy (Yardi).
    `source` = which surface produced the lead (scorecard | space-demand | conversion)."""
    if company_id is None and hmy is None:
        raise ValueError("company_id or hmy required")
    act = (action or "note").strip().lower()
    if act not in _ACTIONS:
        act = "note"
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "company_id": int(company_id) if company_id is not None else None,
           "hmy": int(hmy) if hmy is not None else None,
           "action": act, "staff": (staff or "").strip() or None,
           "note": (note or "").strip() or None, "market": market,
           "source": source, "opportunity_tier": opportunity_tier,
           "rea_score": rea_score, "win_prob": win_prob,
           "conversion_probability": conversion_probability}
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


def history(company_id: int | None = None, hmy: int | None = None, limit: int = 200) -> dict:
    recs = _read()
    if company_id is not None:
        recs = [r for r in recs if r.get("company_id") == int(company_id)]
    if hmy is not None:
        recs = [r for r in recs if r.get("hmy") == int(hmy)]
    recs = list(reversed(recs))[:max(1, int(limit))]
    return {"contract": "signal-desk-brokerage", "view": "lead-history",
            "count": len(recs), "events": recs}


def summary() -> dict:
    """Pursuit coverage + a first-cut uplift read: won-rate among pursued leads grouped by
    the opportunity_tier they carried at contact time (needs enough resolved events to be
    meaningful — reported with counts so it isn't over-read)."""
    recs = _read()
    by_action: dict[str, int] = {}
    tier_outcomes: dict[str, dict] = {}
    for r in recs:
        by_action[r.get("action")] = by_action.get(r.get("action"), 0) + 1
        t = r.get("opportunity_tier") or "unknown"
        d = tier_outcomes.setdefault(t, {"events": 0, "won": 0, "lost": 0})
        d["events"] += 1
        if r.get("action") == "won":
            d["won"] += 1
        elif r.get("action") == "lost":
            d["lost"] += 1
    return {"contract": "signal-desk-brokerage", "view": "lead-summary",
            "total_events": len(recs), "actions": by_action,
            "by_opportunity_tier": tier_outcomes,
            "note": "Uplift becomes meaningful once enough won/lost events accrue per tier."}
