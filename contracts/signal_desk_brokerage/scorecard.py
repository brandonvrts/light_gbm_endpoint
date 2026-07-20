"""Company-centric brokerage scorecard (broker-free). Ranks COMPANIES by multiple
grounded probabilities/signals, all from rea.CompanyScoreSnapshot (live, per company):
REA score + decile, WinProb (live P(win) = cohort base x concept lift), ExpectedValue,
scoop momentum (90d count + recency), lease proximity, and the LLM lifecycle read.

No broker feature anywhere - this answers "which COMPANY is worth pursuing", which is
exactly what the deal-win (broker-conditioned) model could not.
"""
from __future__ import annotations
import math
from core import db

_SORT = {"score": "c.Score", "winprob": "c.WinProb", "value": "c.ExpectedValue",
         "scoops": "c.scoop_count_90d", "recency": "c.scoopRecency", "lease": "c.months_until_expiration"}


def _f(x):
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def _i(x):
    v = _f(x)
    return None if v is None else int(v)


_COLS = ("c.CompanyID, co.Name AS company, c.Market, c.Score, c.Decile, c.WinProb, "
         "c.WinProbBase AS cohort_win_rate, c.WinProbCohort AS cohort, c.WinProbLift AS concept_lift, "
         "c.WinProbLiftTags AS lift_tags, c.ExpectedValue, c.scoop_count_90d, c.scoopRecency, "
         "c.last_scoop_at, c.months_until_expiration, c.inferenceLifecycle, c.HasYardiPresence")


def _row(r) -> dict:
    return {
        "company_id": _i(r.get("CompanyID")), "company": r.get("company"), "market": r.get("Market"),
        "rea_score": round(_f(r.get("Score")) or 0, 1), "decile": _i(r.get("Decile")),
        "win_prob": round(_f(r.get("WinProb")) or 0, 4),
        "cohort_win_rate": round(_f(r.get("cohort_win_rate")) or 0, 4), "cohort": r.get("cohort"),
        "concept_lift": round(_f(r.get("concept_lift")) or 1, 3), "lift_tags": r.get("lift_tags"),
        "expected_value": round(_f(r.get("ExpectedValue")) or 0, 0),
        "scoop_count_90d": _i(r.get("scoop_count_90d")), "scoop_recency": round(_f(r.get("scoopRecency")) or 0, 3),
        "last_scoop_at": r.get("last_scoop_at"),
        "months_to_lease_end": _i(r.get("months_until_expiration")),
        "lifecycle": r.get("inferenceLifecycle"), "has_yardi": bool(r.get("HasYardiPresence")),
    }


def _fill_lease_months(rows: list[dict]) -> None:
    """Phase 2.4: rea.CompanyScoreSnapshot.months_until_expiration is NULL for ~99% of
    companies (the nightly job never populated it). Fill it at serve time for rows that
    link (via ProspectCompanyLink) to a Yardi commercial lease with a future end date
    (~138 of the scored companies have one). Mutates rows in place; never raises."""
    need = [r["company_id"] for r in rows
            if r.get("company_id") is not None and r.get("months_to_lease_end") is None]
    if not need:
        return
    try:
        ids = ",".join(str(int(c)) for c in need)
        df = db.query(
            f"SELECT l.CompanyID, MIN(a.dtEnd) AS soonest "
            f"FROM [{db.DB_MAIN}].rea.ProspectCompanyLink l "
            f"JOIN [{db.DB_MAIN}].rea.vw_Deal d ON d.commProspectId = l.hMy "
            f"JOIN [{db.DB_MAIN}].dbo.CommAmendmentSnapshot a ON a.hTenant = d.tenantId "
            f"WHERE l.CompanyID IN ({ids}) AND a.dtEnd > GETDATE() "
            f"GROUP BY l.CompanyID")
    except Exception:
        return
    if df.empty:
        return
    import datetime as _dt
    today = _dt.date.today()
    by_id = {}
    for r in df.to_dict("records"):
        d = r["soonest"]
        try:
            d = d.date() if hasattr(d, "date") else _dt.date.fromisoformat(str(d)[:10])
            by_id[int(r["CompanyID"])] = round((d - today).days / 30.44, 1)
        except Exception:
            continue
    for row in rows:
        cid = row.get("company_id")
        if row.get("months_to_lease_end") is None and cid in by_id:
            row["months_to_lease_end"] = by_id[cid]
            row["lease_end_source"] = "yardi-amendment (serve-time fill)"


def scorecard_for_prospect(hmy) -> dict | None:
    """The company scorecard row for ONE prospect: hMy -> CompanyID via rea.ProspectCompanyLink,
    then that company's live rea.CompanyScoreSnapshot row. None if unlinked (~54% of prospects
    have no ZI company) or no score row."""
    try:
        df = db.query(
            f"SELECT TOP 1 {_COLS} FROM [{db.DB_MAIN}].rea.ProspectCompanyLink l "
            f"JOIN [{db.DB_MAIN}].rea.CompanyScoreSnapshot c ON c.CompanyID = l.CompanyID "
            f"LEFT JOIN [{db.DB_MAIN}].zi.Company co ON co.CompanyID = c.CompanyID "
            f"WHERE l.hMy = ? AND l.CompanyID IS NOT NULL ORDER BY c.Score DESC", (int(hmy),))
    except Exception:
        return None
    if df.empty:
        return None
    row = _row(df.iloc[0].to_dict())
    _fill_lease_months([row])
    return row


def company_scorecard(sort: str = "score", limit: int = 50, market: str | None = None) -> dict:
    col = _SORT.get((sort or "score").lower(), "c.Score")
    direction = "ASC" if sort == "lease" else "DESC"   # lease: soonest expiry first
    # Tiebreak on the PK, never the sort column: "ORDER BY c.Score DESC, c.Score DESC"
    # is a duplicate-column error (SQL Server 169) and 500'd every default-sort call.
    tiebreak = "c.Score DESC, c.CompanyID DESC" if col != "c.Score" else "c.CompanyID DESC"
    params: tuple = ()
    where = "WHERE c.Score IS NOT NULL"
    if market:
        where += " AND c.Market = ?"
        params = (market,)
    df = db.query(
        f"SELECT TOP {int(limit)} {_COLS} "
        f"FROM [{db.DB_MAIN}].rea.CompanyScoreSnapshot c "
        f"LEFT JOIN [{db.DB_MAIN}].zi.Company co ON co.CompanyID = c.CompanyID "
        f"{where} ORDER BY {col} {direction}, {tiebreak}", params)
    rows = [_row(r) for r in (df.to_dict("records") if not df.empty else [])]
    _fill_lease_months(rows)
    return {
        "contract": "signal-desk-brokerage", "view": "company-scorecard", "sort": sort,
        "count": len(rows), "companies": rows,
        "lenses": ["rea_score (recency-weighted triage rank)", "win_prob (live P(win) = cohort x concept lift)",
                   "cohort_win_rate (size/vertical baseline)", "expected_value ($ = win_prob x cohort value)",
                   "scoop momentum (90d count + recency)", "months_to_lease_end (timing)", "lifecycle (LLM read)"],
        "note": "Company-centric, broker-free. All fields are live from rea.CompanyScoreSnapshot. "
                "Sort by: score | winprob | value | scoops | recency | lease.",
    }
