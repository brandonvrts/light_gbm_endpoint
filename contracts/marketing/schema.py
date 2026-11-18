"""marketing contract metadata. AGGREGATE mode: caller selects a cohort (concept
vertical) or industry; the service returns aggregate predictions/scores over that
segment (counts, mean score/win-prob, decile distribution, cohort win-rate).
"""
from __future__ import annotations

NAME = "marketing"
FEATURE_MODE = "aggregate"   # selectable cohort/industry rollups, not per-entity scoring
VARIANTS = ["cohort", "industry"]


def describe() -> dict:
    return {
        "name": NAME, "feature_mode": FEATURE_MODE, "variants": VARIANTS,
        "input": {"variant": "cohort|industry", "key": "concept key (e.g. healthcare-vertical) "
                                                       "or industry name (e.g. Banking)"},
        "output": {"segment": "key", "company_count": "int", "avg_score": "float",
                   "avg_win_prob": "float", "decile_distribution": "[{decile,count}]",
                   "by_market": "[{market,count,avg_score}]",
                   "cohort_win_rates": "[{vertical,size_band,deal_count,win_rate}]"},
        "space_demand_views": {
            "GET /marketing/space-demand": "per-cohort rollup over the spacedemand ML bundle "
                                           "(expected space-need events ~180d, hot counts)",
            "GET /marketing/space-demand/leads": "ranked leads inside one cohort "
                                                 "(cohort=size=mid|vertical=healthcare)",
            "cohort_vocab": "canonical size bands x verticals (contracts/marketing/cohort.py, "
                            "byte-parity with marketing CohortVocab.cs / rea.DealPattern)"},
    }
