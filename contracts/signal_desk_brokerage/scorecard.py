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
    return None if df.empty else _row(df.iloc[0].to_dict())


def company_scorecard(sort: str = "score", limit: int = 50, market: str | None = None) -> dict:
    col = _SORT.get((sort or "score").lower(), "c.Score")
    direction = "ASC" if sort == "lease" else "DESC"   # lease: soonest expiry first
    params: tuple = ()
    where = "WHERE c.Score IS NOT NULL"
    if market:
        where += " AND c.Market = ?"
        params = (market,)
    df = db.query(
        f"SELECT TOP {int(limit)} {_COLS} "
        f"FROM [{db.DB_MAIN}].rea.CompanyScoreSnapshot c "
        f"LEFT JOIN [{db.DB_MAIN}].zi.Company co ON co.CompanyID = c.CompanyID "
        f"{where} ORDER BY {col} {direction}, c.Score DESC", params)
    rows = [_row(r) for r in (df.to_dict("records") if not df.empty else [])]
    return {
        "contract": "signal-desk-brokerage", "view": "company-scorecard", "sort": sort,
        "count": len(rows), "companies": rows,
        "lenses": ["rea_score (recency-weighted triage rank)", "win_prob (live P(win) = cohort x concept lift)",
                   "cohort_win_rate (size/vertical baseline)", "expected_value ($ = win_prob x cohort value)",
                   "scoop momentum (90d count + recency)", "months_to_lease_end (timing)", "lifecycle (LLM read)"],
        "note": "Company-centric, broker-free. All fields are live from rea.CompanyScoreSnapshot. "
                "Sort by: score | winprob | value | scoops | recency | lease.",
    }
