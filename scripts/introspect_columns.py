"""Phase 0b: dump exact columns for the objects the data layer will query, so
Phase 1 builds against real schema. Read-only.
"""
from __future__ import annotations
from pathlib import Path
import pyodbc

HERE = Path(__file__).resolve().parent.parent


def load_env(path: Path) -> dict:
    env = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip("'").strip('"')
    return env


def connect(env):
    cs = (f"DRIVER={{{env['GBME_DB_DRIVER']}}};SERVER={env['GBME_DB_SERVER']};"
          f"PORT={env.get('GBME_DB_PORT','1433')};UID={env['GBME_DB_USERNAME']};"
          f"PWD={env['GBME_DB_PASSWORD']};TDS_Version={env.get('GBME_DB_TDS_VERSION','7.4')};")
    return pyodbc.connect(cs, autocommit=True, timeout=10)


def cols(cur, db, schema, table):
    cur.execute(f"""SELECT c.name, t.name FROM [{db}].sys.columns c
                    JOIN [{db}].sys.types t ON t.user_type_id=c.user_type_id
                    JOIN [{db}].sys.objects o ON o.object_id=c.object_id
                    JOIN [{db}].sys.schemas s ON s.schema_id=o.schema_id
                    WHERE s.name='{schema}' AND o.name='{table}' ORDER BY c.column_id""")
    rows = cur.fetchall()
    print(f"\n--- {db}.{schema}.{table}  ({len(rows)} cols) ---")
    print(", ".join(f"{r[0]}:{r[1]}" for r in rows))


def sample(cur, sql, label, n=10):
    try:
        cur.execute(sql)
        rows = cur.fetchall()
        print(f"\n--- {label} ({len(rows)}) ---")
        for r in rows[:n]:
            print("   ", tuple(r))
    except Exception as e:
        print(f"\n--- {label}: ERR {str(e)[:120]} ---")


def main():
    env = load_env(HERE / ".env")
    cur = connect(env).cursor()
    im = env.get("GBME_DB_DATABASE", "InsightMart")

    for (sch, tbl) in [("rea", "ProspectScore"), ("rea", "ProspectFeatureSnapshot"),
                       ("rea", "CompanyFeatureSnapshot"), ("rea", "CompanyScoreSnapshot"),
                       ("rea", "ProspectLabelSnapshot"), ("rea", "Entity"),
                       ("rea", "EntityConcept"), ("rea", "Concept"),
                       ("mkt", "CohortCatalog"), ("mkt", "SpaceInferenceFit"),
                       ("mkt", "CohortOccupancyStat"), ("zi", "Company"), ("zi", "Industry")]:
        try:
            cols(cur, im, sch, tbl)
        except Exception as e:
            print(f"\n--- {im}.{sch}.{tbl}: ERR {str(e)[:120]} ---")

    # marketing cohort/industry option sets
    sample(cur, f"SELECT TOP 25 ConceptKey, COUNT(*) FROM [{im}].rea.EntityConcept GROUP BY ConceptKey ORDER BY 2 DESC",
           "rea.EntityConcept ConceptKey distribution", 25)
    sample(cur, f"SELECT TOP 20 Name FROM [{im}].zi.Industry ORDER BY Name", "zi.Industry sample", 20)
    sample(cur, f"SELECT TOP 10 * FROM [{im}].mkt.CohortCatalog", "mkt.CohortCatalog sample", 10)

    # multifamily SP result columns (best-effort via describe_first_result_set)
    mf = env.get("GBME_DB_DATABASE_MF", "InsightMart_Test")
    sample(cur, f"""SELECT name, system_type_name FROM [{mf}].sys.dm_exec_describe_first_result_set(
                    N'EXEC dbo.sp_GetTenantFeaturesByScode @ScodeList=N''t0000000'', @AsOfDate=NULL', NULL, 0)""",
           f"{mf}.sp_GetTenantFeaturesByScode result columns", 80)

    print("\nDONE")


if __name__ == "__main__":
    main()
