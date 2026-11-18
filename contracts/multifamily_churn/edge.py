"""Edge list with hysteresis — the stable fight-for roster (battery fix, 2026-07-22).

The naive edge definition (recompute p >= 0.20 membership every run) reshuffled monthly
(Jaccard 0.11–0.33): about half healthy resolution, half FLAPPING — churn probability
dipping just under the gate with nothing resolved, then bouncing back. This module makes
membership sticky:

  ENTER  when churn p >= 0.20 (no notice filed)
  STAY   until something actually resolves —
           renewed (lease end pushed out >60d), gave notice, moved out — or risk
           genuinely subsides (p < 0.15)
  WATCH  band 0.15 <= p < 0.20: not members yet, but conversations can start early
         (the battery showed members resolve within weeks of entering)

fight_for is decided ONCE at entry (stayer-type trait score vs. the roster median at
that moment, duration bundle v0.1) and stays fixed until exit — the PM instruction for
a resident must not flip week to week.

State lives in monitoring/edge_roster.json (current members) and every enter/exit is
appended to monitoring/edge_roster_log.jsonl with a reason — an auditable history that
future models can learn from. The SSMS seam stays read-only.
"""
from __future__ import annotations

import json
import threading
import time
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from contracts.multifamily_churn import cohorts, data

ENTER_P = 0.20
EXIT_P = 0.15
WATCH_P = 0.15
RENEWAL_PUSH_DAYS = 60
# Pinned 2026-11-18: the ENTER_P/EXIT_P gates were tuned on the h60 ranker (the serving
# default until 2026-11-16). On v8.1 (compressed renewal scores, median ~0.30) nearly the
# whole roster clears 0.20, so following the default would enroll ~640 residents.
# Keep h60 until the gates are re-derived for v8.
EDGE_BUNDLE = "v7-occupancy-h60"

ROOT = Path(__file__).resolve().parent.parent.parent
_STATE_F = ROOT / "monitoring" / "edge_roster.json"
_LOG_F = ROOT / "monitoring" / "edge_roster_log.jsonl"
_DUR_BUNDLE = ROOT / "bundles" / "multifamily_duration" / "v0.1-jan22-backtest" / "model.joblib"
_LOCK = threading.Lock()
_DUR_CACHE: dict = {}


def _duration_bundle():
    hit = _DUR_CACHE.get("b")
    if hit:
        return hit
    b = joblib.load(_DUR_BUNDLE)
    _DUR_CACHE["b"] = b
    return b


def _frame_rows(as_of: str | None) -> tuple[dict, str, dict]:
    """Current population as {scode: row-dict} with churn p, notice, lease timing and
    the stayer-type trait score attached."""
    df, cb = cohorts._scored_frame(None, as_of, version=EDGE_BUNDLE)
    asof = data._safe_asof(as_of)
    if df.empty:
        return {}, asof, {"churn": cb.get("version")}
    b = _duration_bundle()
    v = b["variants"]["trait"]
    X = pd.DataFrame({c: pd.to_numeric(df.get(c), errors="coerce") for c in v["features"]})
    df = df.copy()
    df["trait"] = np.clip(v["model"].predict(X), 0, b["cap_months"])
    df["trait_pctile"] = df["trait"].rank(pct=True) * 100
    rows = {}
    for r in df.to_dict("records"):
        sc = str(r.get("SCODE", "")).strip().lower()
        if not sc:
            continue
        rows[sc] = {
            "scode": str(r.get("SCODE", "")).strip(),
            "name": r.get("FullName"),
            "unit": (str(r.get("UnitCode")).strip() if r.get("UnitCode") is not None else None),
            "property_id": (int(r["PropertyId"]) if pd.notna(r.get("PropertyId")) else None),
            "p": float(r["p_churn"]),
            "notice": bool(pd.to_numeric(pd.Series([r.get("HasNotice")]), errors="coerce").fillna(0).iloc[0] == 1),
            "lease_to": (str(r.get("LeaseToDate") or "")[:10] or None),
            "d2e": (int(r["_d2e"]) if pd.notna(r.get("_d2e")) else None),
            "trait": round(float(r["trait"]), 1),
            "trait_pctile": round(float(r["trait_pctile"]), 0),
            "monthly_rent": (round(float(r["CurrentRent"]), 2)
                             if pd.notna(r.get("CurrentRent")) and float(r.get("CurrentRent") or 0) > 0
                             else (round(float(r["ActiveRent"]), 2)
                                   if pd.notna(r.get("ActiveRent")) else None)),
        }
    return rows, asof, {"churn": cb.get("version"),
                        "duration": "v0.1-jan22-backtest (trait variant)"}


def _renewed(member: dict, row: dict) -> bool:
    old, new = member.get("lease_to_at_entry"), row.get("lease_to")
    if not old or not new:
        return False
    try:
        return (date.fromisoformat(new) - date.fromisoformat(old)).days > RENEWAL_PUSH_DAYS
    except ValueError:
        return False


def apply_hysteresis(state: dict, rows: dict, asof: str) -> tuple[dict, list[dict]]:
    """PURE decision function: (prior members, current population) -> (new members,
    enter/exit events). No I/O — this is what the offline tests exercise."""
    events: list[dict] = []
    new_state: dict = {}

    def _exit(sc: str, m: dict, reason: str, row: dict | None):
        events.append({"ts": asof, "event": "exit", "scode": m.get("scode") or sc,
                       "reason": reason, "entered": m.get("entered"),
                       "fight_for": m.get("fight_for"),
                       "p_now": (round(row["p"], 3) if row else None)})

    for sc, m in state.items():
        row = rows.get(sc)
        if row is None:
            _exit(sc, m, "moved_out", None)
        elif row["notice"]:
            _exit(sc, m, "gave_notice", row)
        elif _renewed(m, row):
            _exit(sc, m, "renewed", row)
        elif row["p"] < EXIT_P:
            _exit(sc, m, "risk_subsided", row)
        else:
            new_state[sc] = {**m, "p_now": round(row["p"], 3), "d2e": row["d2e"],
                             "lease_to": row["lease_to"]}

    entrants = []
    for sc, row in rows.items():
        if sc in state or sc in new_state:
            continue
        if not row["notice"] and row["p"] >= ENTER_P:
            entrants.append((sc, row))
    if entrants:
        # fight_for decided ONCE, at entry: entrant's trait vs the median trait of the
        # roster as it stands this cycle (members + entrants) — then frozen
        pool = [m.get("trait_at_entry") for m in new_state.values()
                if m.get("trait_at_entry") is not None] + [r["trait"] for _, r in entrants]
        med = float(np.median(pool)) if pool else 0.0
        for sc, row in entrants:
            member = {"scode": row["scode"], "entered": asof,
                      "p_at_entry": round(row["p"], 3), "p_now": round(row["p"], 3),
                      "trait_at_entry": row["trait"], "trait_pctile_at_entry": row["trait_pctile"],
                      "lease_to_at_entry": row["lease_to"], "lease_to": row["lease_to"],
                      "d2e": row["d2e"], "fight_for": bool(row["trait"] >= med)}
            new_state[sc] = member
            events.append({"ts": asof, "event": "enter", "scode": row["scode"],
                           "p": round(row["p"], 3), "trait": row["trait"],
                           "fight_for": member["fight_for"]})
    return new_state, events


def _load_state() -> dict:
    if _STATE_F.exists():
        try:
            return json.loads(_STATE_F.read_text())
        except Exception:  # noqa: BLE001
            return {"as_of": None, "members": {}}
    return {"as_of": None, "members": {}}


def _save(state: dict, events: list[dict]) -> None:
    _STATE_F.parent.mkdir(parents=True, exist_ok=True)
    _STATE_F.write_text(json.dumps(state, indent=1))
    if events:
        with _LOG_F.open("a") as fh:
            for e in events:
                fh.write(json.dumps(e) + "\n")


def edge_list(as_of: str | None = None, persist: bool = True) -> dict:
    """The stable edge roster. Same-day calls are idempotent; state only moves forward
    in time (a historical as_of never rewrites the roster — pass persist=False to
    preview)."""
    with _LOCK:
        rows, asof, models = _frame_rows(as_of)
        stored = _load_state()
        if stored.get("as_of") and asof < stored["as_of"]:
            persist = False                     # never rewind the roster
        prior = stored.get("members", {})
        members, events = apply_hysteresis(prior, rows, asof)
        if persist:
            _save({"as_of": asof, "members": members}, events)

    out_members = []
    for sc, m in members.items():
        row = rows.get(sc, {})
        try:
            days_on = (date.fromisoformat(asof) - date.fromisoformat(m["entered"])).days
        except (ValueError, TypeError):
            days_on = None
        out_members.append({
            "scode": m["scode"], "full_name": row.get("name"), "unit": row.get("unit"),
            "property": data.PROPERTIES.get(row.get("property_id")),
            "fight_for": m["fight_for"],
            "churn_probability": round(row.get("p", m.get("p_now", 0.0)), 3),
            "p_at_entry": m["p_at_entry"],
            "stayer_trait_months": m["trait_at_entry"],
            "stayer_trait_pctile": m.get("trait_pctile_at_entry"),
            "monthly_rent": row.get("monthly_rent"),
            "lease_to": row.get("lease_to"), "days_to_lease_end": row.get("d2e"),
            "entered": m["entered"], "days_on_list": days_on})
    out_members.sort(key=lambda x: (not x["fight_for"], -(x["churn_probability"] or 0)))

    watch = [{"scode": r["scode"], "full_name": r["name"], "unit": r["unit"],
              "property": data.PROPERTIES.get(r["property_id"]),
              "churn_probability": round(r["p"], 3),
              "days_to_lease_end": r["d2e"]}
             for sc, r in rows.items()
             if (not r["notice"]) and WATCH_P <= r["p"] < ENTER_P and sc not in members]
    watch.sort(key=lambda x: -x["churn_probability"])

    fight = [m for m in out_members if m["fight_for"]]
    summary = (
        f"The edge roster holds {len(out_members)} residents ({len(fight)} marked fight-for) — "
        f"membership is sticky: residents enter at {ENTER_P:.2f} churn risk and leave only when "
        f"something resolves (renewal, notice, move-out) or risk falls below {EXIT_P:.2f}. "
        f"{len(watch)} more sit in the {WATCH_P:.2f}–{ENTER_P:.2f} watch band. "
        f"{sum(1 for e in events if e['event'] == 'exit')} exit(s) and "
        f"{sum(1 for e in events if e['event'] == 'enter')} entry(ies) this refresh.")

    return {"contract": "multifamily-churn", "view": "edge-list", "as_of": asof,
            "models": models,
            "policy": {"enter_p": ENTER_P, "exit_p": EXIT_P, "watch_band": [WATCH_P, ENTER_P],
                       "renewal_push_days": RENEWAL_PUSH_DAYS,
                       "note": "membership and fight_for are sticky by design (hysteresis) — "
                               "the battery showed the naive monthly recompute reshuffled "
                               "the list (Jaccard 0.11–0.33)"},
            "n": len(out_members), "fight_for_n": len(fight),
            "members": out_members, "watch": watch,
            "changes_this_refresh": events, "summary": summary}


def roster_history(limit: int = 200) -> dict:
    """The enter/exit event log (auditable history of the roster)."""
    events = []
    if _LOG_F.exists():
        for line in _LOG_F.read_text().splitlines():
            try:
                events.append(json.loads(line))
            except Exception:  # noqa: BLE001
                continue
    return {"contract": "multifamily-churn", "view": "edge-roster-history",
            "count": len(events), "events": list(reversed(events))[:max(1, int(limit))]}
