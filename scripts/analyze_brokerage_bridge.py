"""ZI<->Yardi bridge coverage analysis (roadmap Phase 2.3). READ-ONLY.

The rea.ProspectCompanyLink bridge (Yardi prospect hMy <-> ZoomInfo CompanyID) links only
~46% of prospects; the unmatched ~54% get no firmographics/scoops and no company scorecard.
Improving coverage is the highest-leverage data lever (SIGNAL_DESK_DATA_VALIDATION.md), but
WRITING new links is pipeline-owner work on the source DB. This script quantifies the gap
and emits a candidate list of unmatched prospects that name-match a ZI company, so the
owner has a concrete review queue. Writes only a local CSV.

Run: PYTHONPATH=. .venv/bin/python scripts/analyze_brokerage_bridge.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core import db  # noqa: E402

OUT = ROOT / "monitoring" / "bridge_candidates.csv"


def main() -> int:
    D = db.DB_MAIN
    total = db.query(f"SELECT COUNT(*) n FROM [{D}].rea.vw_ProspectEnriched").iloc[0]["n"]
    bridged = db.query(
        f"SELECT COUNT(DISTINCT hMy) n FROM [{D}].rea.ProspectCompanyLink "
        f"WHERE CompanyID IS NOT NULL").iloc[0]["n"]
    print(f"prospects={total} bridged={bridged} coverage={bridged/total:.1%} "
          f"unmatched={total-bridged}")

    # unmatched prospects that have a name -> fuzzy exact-ish name match to a ZI company
    cand = db.query(f"""
        SELECT TOP 500 e.hMy, e.display_name, e.market, z.CompanyID, z.Name AS zi_name
        FROM [{D}].rea.vw_ProspectEnriched e
        JOIN [{D}].zi.Company z
          ON LOWER(LTRIM(RTRIM(z.Name))) = LOWER(LTRIM(RTRIM(e.display_name)))
        WHERE NOT EXISTS (SELECT 1 FROM [{D}].rea.ProspectCompanyLink l
                          WHERE l.hMy = e.hMy AND l.CompanyID IS NOT NULL)
        ORDER BY e.display_name""")
    print(f"exact-name unmatched candidates (auto-linkable): {len(cand)}")
    if not cand.empty:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        cand.to_csv(OUT, index=False)
        print(f"  -> wrote review queue: {OUT}")
        print(cand.head(10).to_string(index=False))
    print("\nHANDOFF: writing rea.ProspectCompanyLink rows is pipeline-owner work on the "
          "source DB. This queue is exact-name matches; a fuzzy/domain pass would recover more.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
