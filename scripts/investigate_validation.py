"""Read-only SSMS grounding for: (1) can Signal Desk be walk-forward validated?
(2) what data supports a marketing cohort-targeting model? No writes."""
from core import db
DB = db.DB_MAIN


def q(sql):
    try:
        df = db.query(sql)
        return df.to_string(index=False) if not df.empty else "(no rows)"
    except Exception as e:  # noqa: BLE001
        return "ERR: " + str(e)[:160]


print("############ SIGNAL DESK (brokerage) — walk-forward feasibility ############")
print("\n-- ProspectLabelSnapshot overview --")
print(q(f"SELECT COUNT(*) rows, COUNT(DISTINCT hMy) entities, MIN(SnapshotDate) dmin, MAX(SnapshotDate) dmax FROM [{DB}].rea.ProspectLabelSnapshot"))
print("\n-- Label x Censored distribution --")
print(q(f"SELECT Label, Censored, COUNT(*) n FROM [{DB}].rea.ProspectLabelSnapshot GROUP BY Label, Censored ORDER BY Label, Censored"))
print("\n-- distinct POSITIVE entities (Label=1) --")
print(q(f"SELECT COUNT(DISTINCT hMy) pos_entities FROM [{DB}].rea.ProspectLabelSnapshot WHERE Label=1"))
print("\n-- positives by quarter (can we make walk-forward folds?) --")
print(q(f"""SELECT DATEPART(YEAR,SnapshotDate) y, DATEPART(QUARTER,SnapshotDate) q,
  COUNT(*) snaps, SUM(CAST(Label AS int)) positives,
  COUNT(DISTINCT CASE WHEN Label=1 THEN hMy END) pos_entities
  FROM [{DB}].rea.ProspectLabelSnapshot GROUP BY DATEPART(YEAR,SnapshotDate),DATEPART(QUARTER,SnapshotDate) ORDER BY y,q"""))
print("\n-- WindowDays values --")
print(q(f"SELECT WindowDays, COUNT(*) n FROM [{DB}].rea.ProspectLabelSnapshot GROUP BY WindowDays ORDER BY WindowDays"))
print("\n-- ProspectFeatureSnapshot range --")
print(q(f"SELECT COUNT(*) rows, COUNT(DISTINCT hMy) entities, MIN(SnapshotDate) dmin, MAX(SnapshotDate) dmax FROM [{DB}].rea.ProspectFeatureSnapshot"))
print("\n-- labeled JOIN (features + label on hMy+date) --")
print(q(f"""SELECT COUNT(*) joined_rows, SUM(CAST(l.Label AS int)) positives,
  COUNT(DISTINCT CASE WHEN l.Label=1 THEN f.hMy END) pos_entities
  FROM [{DB}].rea.ProspectFeatureSnapshot f
  JOIN [{DB}].rea.ProspectLabelSnapshot l ON l.hMy=f.hMy AND l.SnapshotDate=f.SnapshotDate"""))
print("\n-- ScoreRun history (is the rules score stale?) --")
print(q(f"SELECT ScoreRunID, ModelVersion, AsOfDate, ProspectsScored FROM [{DB}].rea.ScoreRun ORDER BY ScoreRunID"))

print("\n\n############ MARKETING — cohort-targeting data ############")
print("\n-- mkt.CohortCatalog coverage + freshness --")
print(q(f"SELECT COUNT(*) rows, COUNT(DISTINCT Vertical) verticals, COUNT(DISTINCT SizeBand) sizebands, SUM(DealCount) deals, SUM(WonCount) won, MAX(SnapshotAt) built FROM [{DB}].mkt.CohortCatalog"))
print("\n-- top cohorts by deal volume --")
print(q(f"SELECT TOP 12 Vertical, SizeBand, DealCount, WonCount, WinRate, ProspectsTagged FROM [{DB}].mkt.CohortCatalog WHERE Vertical IS NOT NULL ORDER BY DealCount DESC"))
print("\n-- EntityConcept cohort sizes (addressable companies per concept) --")
print(q(f"SELECT ConceptKey, COUNT(DISTINCT EntityId) entities FROM [{DB}].rea.EntityConcept WHERE ConceptKey LIKE '%vertical%' OR ConceptKey IN ('growth-mode','lease-expiring-12mo','expanding-footprint','distress','has-active-deal') GROUP BY ConceptKey ORDER BY entities DESC"))
print("\n-- CompanyScoreSnapshot: WinProb / ExpectedValue / lease-timing coverage --")
print(q(f"SELECT COUNT(*) rows, COUNT(WinProb) winprob_nn, COUNT(ExpectedValue) ev_nn, COUNT(months_until_expiration) mexp_nn, COUNT(DISTINCT Market) markets, CAST(AVG(WinProb) AS DECIMAL(6,3)) avg_winprob FROM [{DB}].rea.CompanyScoreSnapshot"))
print("\n-- ExpectedValue sample (is there $ to rank by yield?) --")
print(q(f"SELECT TOP 5 CompanyID, Market, Score, Decile, WinProb, ExpectedValue, months_until_expiration FROM [{DB}].rea.CompanyScoreSnapshot WHERE ExpectedValue IS NOT NULL ORDER BY ExpectedValue DESC"))
print("\n-- industries present (zi_primary_industry) --")
print(q(f"SELECT COUNT(*) rows, COUNT(zi_primary_industry) with_industry, COUNT(DISTINCT zi_primary_industry) industries FROM [{DB}].rea.vw_ProspectEnriched"))
print("\nDONE")
