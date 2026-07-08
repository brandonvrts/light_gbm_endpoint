"""Phase 0 EVAL: prove real, per-app data ingestion against the live DB end-to-end.
PASS criteria: multifamily SP returns feature rows for real SCODEs; brokerage feature
snapshot is queryable per company with history; marketing cohort/industry aggregates query.
Read-only.
"""
from __future__ import annotations
from pathlib import Path
from datetime import date
import pyodbc

HERE = Path(__file__).resolve().parent.parent


def load_env(path):
    env = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip("'").strip('"')
    return env


def connect(env):
    cs = (f"DRIVER={{{env['GBME_DB_DRIVER']}}};SERVER={env['GBME_DB_SERVER']};"
          f"PORT={env.get('GBME_DB_PORT','1433')};UID={env['GBME_DB_USERNAME']};"
          f"PWD={env['GBME_DB_PASSWORD']};TDS_Version={env.get('GBME_DB_TDS_VERSION','7.4')};")
    return pyodbc.connect(cs, autocommit=True, timeout=15)


def main():
    env = load_env(HERE / ".env")
    im = env.get("GBME_DB_DATABASE", "InsightMart")
    mf = env.get("GBME_DB_DATABASE_MF", "InsightMart_Test")
    yardi = env.get("GBME_DB_YARDI", "Yardi7_Reporting")
    cur = connect(env).cursor()
    results = {}

    print("=" * 70, "\nMULTIFAMILY\n", "=" * 70)
    try:
        cur.execute(f"SELECT TOP 5 SCODE, HPROPERTY FROM [{yardi}].dbo.TENANT "
                    f"WHERE HPROPERTY IN (62,72,82,26) AND SCODE IS NOT NULL ORDER BY NEWID()")
        scodes = [r[0].strip() for r in cur.fetchall()]
        print("sample in-scope SCODEs:", scodes)
        if scodes:
            lst = ",".join(scodes)
            cur.execute("EXEC [" + mf + "].dbo.sp_GetTenantFeaturesByScode @ScodeList=?, @AsOfDate=?",
                        lst, date.today())
            cols = [c[0] for c in cur.description]
            rows = cur.fetchall()
            print(f"sp_GetTenantFeaturesByScode -> {len(cols)} cols, {len(rows)} rows")
            print("columns:", ", ".join(cols))
            results["multifamily"] = "PASS" if cols and rows else "WARN (cols but no rows)"
        else:
            results["multifamily"] = "FAIL (no SCODEs)"
    except Exception as e:
        print("MF ERR:", str(e)[:240]); results["multifamily"] = "FAIL"

    print("\n", "=" * 70, "\nBROKERAGE (Signal Desk)\n", "=" * 70)
    try:
        cur.execute(f"SELECT TOP 1 hMy FROM [{im}].rea.ProspectFeatureSnapshot "
                    f"GROUP BY hMy ORDER BY COUNT(*) DESC")
        hmy = cur.fetchone()[0]
        cur.execute(f"SELECT COUNT(*), MIN(SnapshotDate), MAX(SnapshotDate) "
                    f"FROM [{im}].rea.ProspectFeatureSnapshot WHERE hMy=?", hmy)
        n, dmin, dmax = cur.fetchone()
        print(f"company hMy={hmy}: {n} historical snapshots, {dmin}..{dmax} (this is the 'run through all historicals' path)")
        cur.execute(f"SELECT TOP 1 Probability, Decile FROM [{im}].rea.ProspectScore WHERE hMy=?", hmy)
        sc = cur.fetchone()
        print(f"current rea.ProspectScore for hMy={hmy}:", tuple(sc) if sc else "none")
        results["brokerage"] = "PASS" if n and n > 0 else "WARN"
    except Exception as e:
        print("BROKERAGE ERR:", str(e)[:240]); results["brokerage"] = "FAIL"

    print("\n", "=" * 70, "\nMARKETING (cohort / industry aggregates)\n", "=" * 70)
    try:
        cur.execute(f"""SELECT TOP 8 Vertical, SizeBand, DealCount, WinRate
                        FROM [{im}].mkt.CohortCatalog WHERE Vertical IS NOT NULL
                        ORDER BY DealCount DESC""")
        print("cohort win-rates (mkt.CohortCatalog):")
        for r in cur.fetchall():
            print("   ", tuple(r))
        cur.execute(f"""SELECT TOP 8 ec.ConceptKey, COUNT(*) n
                        FROM [{im}].rea.EntityConcept ec
                        WHERE ec.ConceptKey LIKE '%vertical%' GROUP BY ec.ConceptKey ORDER BY n DESC""")
        print("industry/vertical cohorts (selectable):")
        for r in cur.fetchall():
            print("   ", tuple(r))
        results["marketing"] = "PASS"
    except Exception as e:
        print("MARKETING ERR:", str(e)[:240]); results["marketing"] = "FAIL"

    print("\n", "=" * 70, "\nVERDICT\n", "=" * 70)
    for k, v in results.items():
        print(f"  {k:12s}: {v}")
    overall = "PASS" if all(v.startswith("PASS") for v in results.values()) else "REVIEW"
    print(f"  OVERALL    : {overall}")


if __name__ == "__main__":
    main()
