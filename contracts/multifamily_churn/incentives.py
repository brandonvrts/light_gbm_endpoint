"""Incentive/concession history surface (read-only).

System of record: Yardi7_Reporting.dbo.TRANS credits (negative STOTALAMOUNT) on the 8
landlord concession accounts — the 2026-07-21 census confirmed every purpose-built
concession table (RMConcession*, ConcessionChargeCodes, ...) is empty and gprhistory
under-counts. Account 40 (third-party subsidy) is INCOME, not a concession — excluded.

TRANS shows CONSUMED concessions only (posted on a lease); offers — accepted AND
declined — live in the outreach log (incentive_* actions). Surfacing both here is what
lets PMs see "what has this resident already been given / offered" next to the edge
list, and lets a future model learn from incentives over time.
"""
from __future__ import annotations

import pandas as pd

from core import db
from contracts.multifamily_churn import data, outreach

# HRETENTIONACCT -> display name (TRANS census 2026-07-21)
ACCOUNTS: dict[int, str] = {
    54: "One Time Concessions",
    50: "Senior Citizen Discount",
    29: "Rent Concessions Monthly",      # ended 2024-10
    53: "Renewal Concession",
    51: "Employee Discount",
    30: "Garage Concession",
    33: "5% Discount",
    56: "Cable Concessions",             # ended 2021-11
}
_ACCT_LIST = ",".join(str(a) for a in ACCOUNTS)


def _events_df(scodes: list[str] | None = None, property_id: int | None = None,
               since: str | None = None, limit: int = 500) -> pd.DataFrame:
    """Concession credit events from TRANS, newest first. Filter by resident scodes
    and/or property; parameterized throughout."""
    where, params = [f"tr.HRETENTIONACCT IN ({_ACCT_LIST})", "tr.STOTALAMOUNT < 0",
                     "ISNULL(tr.VOID, 0) = 0", "ISNULL(tr.VOIDED, 0) = 0"], []
    pids = ",".join(str(int(p)) for p in data.DEFAULT_PROPERTY_IDS)
    where.append(f"t.HPROPERTY IN ({pids})")
    if scodes:
        ph = ",".join("?" for _ in scodes)
        where.append(f"t.SCODE IN ({ph})")
        params.extend([str(s).strip() for s in scodes])
    if property_id is not None:
        where.append("t.HPROPERTY = ?")
        params.append(int(property_id))
    if since:
        where.append("tr.UPOSTDATE >= ?")
        params.append(str(since)[:10])
    sql = (f"SELECT TOP {int(limit)} t.SCODE, t.HPROPERTY, tr.UPOSTDATE, "
           f"tr.HRETENTIONACCT acct, -tr.STOTALAMOUNT amount, tr.SNOTES note "
           f"FROM [{db.DB_YARDI}].dbo.TRANS tr "
           f"JOIN [{db.DB_YARDI}].dbo.TENANT t ON t.HMYPERSON = tr.HPERSON "
           f"WHERE {' AND '.join(where)} ORDER BY tr.UPOSTDATE DESC")
    return db.query(sql, tuple(params))


def _event_row(r: dict) -> dict:
    acct = int(r["acct"]) if pd.notna(r.get("acct")) else None
    return {"date": str(r.get("UPOSTDATE") or "")[:10] or None,
            "account": acct, "account_name": ACCOUNTS.get(acct, "unknown"),
            "amount": round(float(r["amount"]), 2) if pd.notna(r.get("amount")) else None,
            "note": (str(r.get("note") or "").strip() or None)}


def for_scodes(scodes: list[str], since: str | None = None,
               per_resident: int = 3) -> dict:
    """Batch: scode -> compact incentive summary (for embedding in the edge-resident
    list). Includes consumed TRANS concessions + logged offers from the outreach log."""
    out: dict[str, dict] = {}
    if not scodes:
        return out
    df = _events_df(scodes=scodes, since=since, limit=2000)
    by_scode: dict[str, list] = {}
    if not df.empty:
        for r in df.to_dict("records"):
            by_scode.setdefault(str(r["SCODE"]).strip().lower(), []).append(_event_row(r))
    offers: dict[str, list] = {}
    for ev in outreach._read():
        if str(ev.get("action", "")).startswith("incentive_"):
            offers.setdefault(str(ev.get("scode") or "").strip().lower(), []).append(ev)
    for s in scodes:
        k = str(s).strip().lower()
        evs = by_scode.get(k, [])
        offs = offers.get(k, [])
        out[k] = {
            "past_concessions_n": len(evs),
            "past_concessions_total": round(sum(e["amount"] or 0 for e in evs), 2),
            "recent_concessions": evs[:max(int(per_resident), 0)],
            "offers_logged_n": len(offs),
            "last_offer": (sorted(offs, key=lambda e: e.get("ts") or "")[-1] if offs else None)}
    return out


def _totals(scode: str | None, property_id: int | None, since: str | None) -> dict:
    """Uncapped per-account aggregate over the SAME filters as the event list — totals
    must never depend on the display `limit`."""
    where, params = [f"tr.HRETENTIONACCT IN ({_ACCT_LIST})", "tr.STOTALAMOUNT < 0",
                     "ISNULL(tr.VOID, 0) = 0", "ISNULL(tr.VOIDED, 0) = 0"], []
    pids = ",".join(str(int(p)) for p in data.DEFAULT_PROPERTY_IDS)
    where.append(f"t.HPROPERTY IN ({pids})")
    if scode:
        where.append("t.SCODE = ?")
        params.append(str(scode).strip())
    if property_id is not None:
        where.append("t.HPROPERTY = ?")
        params.append(int(property_id))
    if since:
        where.append("tr.UPOSTDATE >= ?")
        params.append(str(since)[:10])
    df = db.query(
        f"SELECT tr.HRETENTIONACCT acct, COUNT(*) n, SUM(-tr.STOTALAMOUNT) dollars "
        f"FROM [{db.DB_YARDI}].dbo.TRANS tr "
        f"JOIN [{db.DB_YARDI}].dbo.TENANT t ON t.HMYPERSON = tr.HPERSON "
        f"WHERE {' AND '.join(where)} GROUP BY tr.HRETENTIONACCT", tuple(params))
    by_acct = {ACCOUNTS.get(int(r["acct"]), "unknown"):
               {"events": int(r["n"]), "dollars": round(float(r["dollars"]), 2)}
               for r in df.to_dict("records")} if not df.empty else {}
    return {"events": sum(v["events"] for v in by_acct.values()),
            "dollars": round(sum(v["dollars"] for v in by_acct.values()), 2),
            "by_account": by_acct}


def history(scode: str | None = None, property_id: int | None = None,
            since: str | None = None, limit: int = 100) -> dict:
    """Standalone incentive-history view: TRANS concession events (+ per-account totals
    over the FULL filtered set, independent of `limit`) and logged incentive offers,
    filterable by resident and/or property."""
    df = _events_df(scodes=[scode] if scode else None, property_id=property_id,
                    since=since, limit=max(int(limit), 1))
    events = [] if df.empty else [
        {**_event_row(r), "scode": str(r["SCODE"]).strip(),
         "property": data.PROPERTIES.get(int(r["HPROPERTY"]))
                     if pd.notna(r.get("HPROPERTY")) else None}
        for r in df.to_dict("records")]
    totals = _totals(scode, property_id, since)
    offers = [ev for ev in outreach._read()
              if str(ev.get("action", "")).startswith("incentive_")
              and (not scode or (ev.get("scode") or "").strip().lower() == scode.strip().lower())]
    return {"contract": "multifamily-churn", "view": "incentive-history",
            "filters": {"scode": scode, "property_id": property_id, "since": since},
            "source": "Yardi TRANS concession credits (consumed) + outreach log (offers)",
            "note": "TRANS records concessions actually posted to a lease; offers — "
                    "including declined ones — exist only in the outreach log "
                    "(action=incentive_offered/accepted/declined).",
            "totals": totals,
            "events_returned": len(events),
            "events": events,
            "logged_offers": list(reversed(offers))[:max(int(limit), 1)]}
