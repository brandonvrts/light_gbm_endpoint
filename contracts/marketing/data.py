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


def _space_demand_by_key() -> dict:
    """Per-cohort 'needs space now' stats from the space-demand ML rollup, keyed by the
    canonical CohortKey (Phase 2 fusion). Best-effort: targeting must keep working from
    the catalog alone if scoring is unavailable."""
    try:
        from contracts.marketing import space_demand as sdm
        roll = sdm.cohort_rollup(market=None, min_scored=1)
        by_key = {}
        for c in roll.get("cohorts", []) + roll.get("verticals", []):
            by_key[c["cohort_key"]] = {
                k: c[k] for k in ("n_scored", "hot_n", "hot_new_n",
                                  "expected_events_180d", "avg_propensity")}
        return {"by_key": by_key, "model_source": roll.get("model_source"),
                "hot_definition": roll.get("hot_definition")}
    except Exception:  # noqa: BLE001
        return {"by_key": {}, "model_source": None, "hot_definition": None}


def cohort_targeting() -> dict:
    """Rank cohorts the marketing team should work. Primary rank: EXPECTED OPPORTUNITY =
    hot NEW space-need leads now (space-demand ML rollup, not already in pipeline) x
    historical win-rate; cohorts without scoop coverage fall back to expected_winnable =
    addressable x win-rate (no data != no demand). Enriched with per-industry scoop
    momentum. Grounded in mkt.CohortCatalog + rea.DealPattern (real won/deals) + the
    spacedemand bundle + zi.Scoop (2022-2026 backfill); no hallucinated $ value
    (DealPattern median rent is unpopulated, so $ EV is intentionally omitted)."""
    from contracts.marketing import cohort as cv
    df = db.query(
        f"SELECT Vertical, SizeBand, DealCount, WonCount, LostCount, WinRate, ProspectsTagged, "
        f"MedianDaysInStage FROM [{db.DB_MAIN}].mkt.CohortCatalog "
        f"WHERE Vertical IS NOT NULL AND SizeBand IS NOT NULL AND ProspectsTagged > 0")
    def _i(x):
        return int(x) if pd.notna(x) else 0

    def _f(x):
        return float(x) if pd.notna(x) else 0.0

    sd = _space_demand_by_key()

    def _fuse(entry: dict, key: str, wr: float) -> dict:
        stats = sd["by_key"].get(key)
        entry["cohort_key"] = key
        entry["needs_space_now"] = stats
        entry["expected_opportunity"] = (
            round(stats["hot_new_n"] * wr, 2) if stats else 0.0)
        return entry

    cohorts = []
    for r in (df.to_dict("records") if not df.empty else []):
        addr, wr, mdis = _i(r.get("ProspectsTagged")), _f(r.get("WinRate")), r.get("MedianDaysInStage")
        cohorts.append(_fuse(
            {"vertical": r["Vertical"], "size_band": r["SizeBand"],
             "addressable": addr, "win_rate": round(wr, 3),
             "deal_count": _i(r.get("DealCount")), "won": _i(r.get("WonCount")),
             "expected_winnable": round(addr * wr, 1),
             "median_days_in_stage": (int(mdis) if pd.notna(mdis) else None)},
            cv.cohort_key(r["SizeBand"], r["Vertical"]), wr))
    cohorts.sort(key=lambda x: (x["expected_opportunity"], x["expected_winnable"]), reverse=True)

    vr = db.query(
        f"SELECT Vertical, SUM(DealCount) deals, SUM(WonCount) won, SUM(ProspectsTagged) addressable, "
        f"CAST(SUM(WonCount)*1.0/NULLIF(SUM(DealCount),0) AS DECIMAL(7,4)) win_rate "
        f"FROM [{db.DB_MAIN}].mkt.CohortCatalog WHERE Vertical IS NOT NULL AND SizeBand IS NOT NULL "
        f"GROUP BY Vertical")
    verticals = []
    for r in (vr.to_dict("records") if not vr.empty else []):
        addr, wr = _i(r.get("addressable")), _f(r.get("win_rate"))
        verticals.append(_fuse(
            {"vertical": r["Vertical"], "addressable": addr, "win_rate": round(wr, 3),
             "deals": _i(r.get("deals")), "won": _i(r.get("won")),
             "expected_winnable": round(addr * wr, 1)},
            cv.cohort_key(None, r["Vertical"]), wr))
    verticals.sort(key=lambda x: (x["expected_opportunity"], x["expected_winnable"]), reverse=True)
    try:
        momentum = scoop_momentum(90)
    except Exception:
        momentum = []
    return {"objective": "hot new space-need leads x historical win-rate "
                         "(fallback: addressable x win-rate) + scoop momentum",
            "source": "mkt.CohortCatalog + spacedemand bundle + rea.ProspectCompanyLink "
                      "+ zi.Scoop (2022-2026)",
            "space_demand_model": sd["model_source"],
            "hot_definition": sd["hot_definition"],
            "cohorts": cohorts, "verticals": verticals, "scoop_momentum_90d": momentum,
            "note": "expected_opportunity = hot NEW space-need leads (ML, top-decile propensity, "
                    "not in pipeline) x historical win-rate — the 'campaign at these now' rank. "
                    "needs_space_now = null means no scoop coverage (no data, NOT no demand); such "
                    "cohorts rank by expected_winnable = addressable x historical win-rate. "
                    "scoop_momentum_90d = recent ZoomInfo scoop volume per industry (heating-up "
                    "signal). $ EV omitted: rea.DealPattern median rent is not populated."}


def _records(df: pd.DataFrame) -> list[dict]:
    return [] if df.empty else df.to_dict("records")
