"""Marketing live data access. Aggregate model outputs over cohorts / industries.

Sources (all real, SSMS): rea.Concept (selectable cohorts), rea.CompanyScoreSnapshot
(per-company Score/Decile/WinProb + ConceptKeys), rea.vw_ProspectEnriched (industry),
mkt.CohortCatalog (deal win-rates by vertical x size band).
"""
from __future__ import annotations
import pandas as pd
from core import db


def list_options() -> dict:
    cohorts = db.query(
        f"SELECT ConceptKey, ConceptLabel, ConceptType FROM [{db.DB_MAIN}].rea.Concept "
        f"WHERE IsActive=1 ORDER BY ConceptType, ConceptKey")
    industries = db.query(
        f"SELECT zi_primary_industry AS industry, COUNT(*) n "
        f"FROM [{db.DB_MAIN}].rea.vw_ProspectEnriched WHERE zi_primary_industry IS NOT NULL "
        f"GROUP BY zi_primary_industry HAVING COUNT(*) >= 2 ORDER BY n DESC")
    sizes = db.query(
        f"SELECT DISTINCT SizeBand FROM [{db.DB_MAIN}].mkt.CohortCatalog WHERE SizeBand IS NOT NULL")
    return {
        "cohorts": [] if cohorts.empty else cohorts.to_dict("records"),
        "industries": [] if industries.empty else industries.to_dict("records"),
        "size_bands": [] if sizes.empty else sizes["SizeBand"].tolist(),
    }


def cohort_aggregate(concept_key: str) -> dict:
    """ConceptKeys (comma-delimited) lives on rea.CompanyFeatureSnapshot; aggregate its
    latest snapshot. Uses EXACT token match (',' + keys + ',' LIKE '%,key,%') so e.g.
    'tech-vertical' does NOT match 'fintech-vertical'."""
    tok = f"%,{concept_key},%"
    latest = (f"SELECT * FROM [{db.DB_MAIN}].rea.CompanyFeatureSnapshot "
              f"WHERE SnapshotDate=(SELECT MAX(SnapshotDate) FROM [{db.DB_MAIN}].rea.CompanyFeatureSnapshot)")
    by_market = db.query(
        f"SELECT Market, COUNT(*) n, AVG(Score) avg_score, AVG(WinProb) avg_win_prob "
        f"FROM ({latest}) s WHERE ',' + s.ConceptKeys + ',' LIKE ? GROUP BY Market", (tok,))
    deciles = db.query(
        f"SELECT Decile, COUNT(*) n FROM ({latest}) s WHERE ',' + s.ConceptKeys + ',' LIKE ? "
        f"GROUP BY Decile ORDER BY Decile", (tok,))
    return {"segment": concept_key, "by_market": _records(by_market),
            "decile_distribution": _records(deciles)}


def industry_aggregate(industry: str) -> dict:
    deciles = db.query(
        f"SELECT Decile, COUNT(*) n FROM [{db.DB_MAIN}].rea.vw_ProspectEnriched "
        f"WHERE zi_primary_industry=? GROUP BY Decile ORDER BY Decile", (industry,))
    summary = db.query(
        f"SELECT COUNT(*) n, AVG(Probability) avg_prob, AVG(CAST(Decile AS float)) avg_decile, "
        f"COUNT(DISTINCT market) markets FROM [{db.DB_MAIN}].rea.vw_ProspectEnriched "
        f"WHERE zi_primary_industry=?", (industry,))
    return {"segment": industry, "summary": _records(summary),
            "decile_distribution": _records(deciles)}


def cohort_win_rates(vertical: str | None = None) -> pd.DataFrame:
    if vertical:
        return db.query(
            f"SELECT Vertical, SizeBand, DealCount, WonCount, LostCount, WinRate, MedianDaysInStage "
            f"FROM [{db.DB_MAIN}].mkt.CohortCatalog WHERE Vertical=? ORDER BY DealCount DESC",
            (vertical,))
    return db.query(
        f"SELECT Vertical, SizeBand, DealCount, WonCount, LostCount, WinRate, MedianDaysInStage "
        f"FROM [{db.DB_MAIN}].mkt.CohortCatalog WHERE Vertical IS NOT NULL ORDER BY DealCount DESC")


def scoop_momentum(window_days: int = 90) -> list[dict]:
    """'Heating up' signal: recent ZoomInfo scoop volume per industry, joined through the
    rea.ProspectCompanyLink bridge (hMy->CompanyID) to zi.Scoop. Dedups (industry, CompanyID)
    so a multi-prospect company is not double-counted. Uses the 2022-2026 scoop backfill."""
    df = db.query(
        f"SELECT ci.industry, COUNT(s.ScoopID) scoops, COUNT(DISTINCT ci.CompanyID) companies "
        f"FROM (SELECT DISTINCT pe.zi_primary_industry AS industry, l.CompanyID "
        f"      FROM [{db.DB_MAIN}].rea.vw_ProspectEnriched pe "
        f"      JOIN [{db.DB_MAIN}].rea.ProspectCompanyLink l ON l.hMy=pe.hMy AND l.CompanyID IS NOT NULL "
        f"      WHERE pe.zi_primary_industry IS NOT NULL) ci "
        f"JOIN [{db.DB_MAIN}].zi.Scoop s ON s.CompanyID=ci.CompanyID "
        f"   AND s.OriginalPublishedDate >= DATEADD(DAY, -{int(window_days)}, GETDATE()) "
        f"GROUP BY ci.industry ORDER BY scoops DESC")
    if df.empty:
        return []
    return [{"industry": r["industry"], "scoops": int(r["scoops"]),
             "companies": int(r["companies"]),
             "scoops_per_company": round(float(r["scoops"]) / max(int(r["companies"]), 1), 2)}
            for r in df.to_dict("records")]


def cohort_targeting() -> dict:
    """Rank cohorts the marketing team should work, by EXPECTED OPPORTUNITY =
    addressable prospects x historical win-rate, enriched with a recent scoop-momentum
    signal per industry. Grounded in mkt.CohortCatalog + rea.DealPattern (real won/deals)
    + zi.Scoop (2022-2026 backfill); no hallucinated $ value (DealPattern median rent is
    unpopulated, so $ EV is intentionally omitted)."""
    df = db.query(
        f"SELECT Vertical, SizeBand, DealCount, WonCount, LostCount, WinRate, ProspectsTagged, "
        f"MedianDaysInStage FROM [{db.DB_MAIN}].mkt.CohortCatalog "
        f"WHERE Vertical IS NOT NULL AND SizeBand IS NOT NULL AND ProspectsTagged > 0")
    def _i(x):
        return int(x) if pd.notna(x) else 0

    def _f(x):
        return float(x) if pd.notna(x) else 0.0

    cohorts = []
    for r in (df.to_dict("records") if not df.empty else []):
        addr, wr, mdis = _i(r.get("ProspectsTagged")), _f(r.get("WinRate")), r.get("MedianDaysInStage")
        cohorts.append({"vertical": r["Vertical"], "size_band": r["SizeBand"],
                        "addressable": addr, "win_rate": round(wr, 3),
                        "deal_count": _i(r.get("DealCount")), "won": _i(r.get("WonCount")),
                        "expected_winnable": round(addr * wr, 1),
                        "median_days_in_stage": (int(mdis) if pd.notna(mdis) else None)})
    cohorts.sort(key=lambda x: x["expected_winnable"], reverse=True)

    vr = db.query(
        f"SELECT Vertical, SUM(DealCount) deals, SUM(WonCount) won, SUM(ProspectsTagged) addressable, "
        f"CAST(SUM(WonCount)*1.0/NULLIF(SUM(DealCount),0) AS DECIMAL(7,4)) win_rate "
        f"FROM [{db.DB_MAIN}].mkt.CohortCatalog WHERE Vertical IS NOT NULL AND SizeBand IS NOT NULL "
        f"GROUP BY Vertical")
    verticals = []
    for r in (vr.to_dict("records") if not vr.empty else []):
        addr, wr = _i(r.get("addressable")), _f(r.get("win_rate"))
        verticals.append({"vertical": r["Vertical"], "addressable": addr, "win_rate": round(wr, 3),
                          "deals": _i(r.get("deals")), "won": _i(r.get("won")),
                          "expected_winnable": round(addr * wr, 1)})
    verticals.sort(key=lambda x: x["expected_winnable"], reverse=True)
    try:
        momentum = scoop_momentum(90)
    except Exception:
        momentum = []
    return {"objective": "addressable x historical win-rate + scoop momentum",
            "source": "mkt.CohortCatalog + rea.ProspectCompanyLink + zi.Scoop (2022-2026)",
            "cohorts": cohorts, "verticals": verticals, "scoop_momentum_90d": momentum,
            "note": "expected_winnable = addressable x historical win-rate (rank to prioritize outreach); "
                    "scoop_momentum_90d = recent ZoomInfo scoop volume per industry (heating-up signal). "
                    "$ EV omitted: rea.DealPattern median rent is not populated."}


def _records(df: pd.DataFrame) -> list[dict]:
    return [] if df.empty else df.to_dict("records")
