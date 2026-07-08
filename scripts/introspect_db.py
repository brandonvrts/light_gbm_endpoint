"""Phase 0 probe: connect read-only to the live SQL Server (SSMS source of truth)
and discover what actually exists for the 3 apps. Builds nothing; prints findings.

Run: . .venv/bin/activate && python scripts/introspect_db.py
"""
from __future__ import annotations
import os
import sys
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


def connect(env: dict):
    cs = (
        f"DRIVER={{{env['GBME_DB_DRIVER']}}};"
        f"SERVER={env['GBME_DB_SERVER']};PORT={env.get('GBME_DB_PORT', '1433')};"
        f"UID={env['GBME_DB_USERNAME']};PWD={env['GBME_DB_PASSWORD']};"
        f"TDS_Version={env.get('GBME_DB_TDS_VERSION', '7.4')};"
    )
    return pyodbc.connect(cs, autocommit=True, timeout=10)


def q(cur, sql):
    cur.execute(sql)
    cols = [c[0] for c in cur.description] if cur.description else []
    return cols, cur.fetchall()


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def list_procs(cur, db, like="sp_Get%"):
    try:
        _, rows = q(cur, f"SELECT name FROM [{db}].sys.procedures WHERE name LIKE '{like}' ORDER BY name")
        print(f"[{db}] stored procs LIKE '{like}': {len(rows)}")
        for r in rows:
            print(f"    - {r[0]}")
    except Exception as e:
        print(f"[{db}] proc list FAILED: {str(e)[:160]}")


def list_objects(cur, db, schema):
    try:
        _, rows = q(cur, f"""SELECT o.name, o.type_desc
                             FROM [{db}].sys.objects o JOIN [{db}].sys.schemas s ON s.schema_id=o.schema_id
                             WHERE s.name='{schema}' AND o.type IN ('U','V') ORDER BY o.name""")
        print(f"[{db}].{schema}: {len(rows)} tables/views")
        for r in rows:
            print(f"    - {schema}.{r[0]} ({r[1]})")
    except Exception as e:
        print(f"[{db}].{schema} FAILED: {str(e)[:160]}")


def rowcount(cur, db, schema, obj):
    try:
        _, rows = q(cur, f"""SELECT SUM(p.rows) FROM [{db}].sys.partitions p
                             JOIN [{db}].sys.objects o ON o.object_id=p.object_id
                             JOIN [{db}].sys.schemas s ON s.schema_id=o.schema_id
                             WHERE s.name='{schema}' AND o.name='{obj}' AND p.index_id IN (0,1)""")
        n = rows[0][0] if rows and rows[0][0] is not None else None
        print(f"    {db}.{schema}.{obj}: {'MISSING' if n is None else f'~{n:,} rows'}")
    except Exception as e:
        print(f"    {db}.{schema}.{obj}: ERR {str(e)[:100]}")


def main():
    env = load_env(HERE / ".env")
    print(f"Connecting to {env['GBME_DB_SERVER']}:{env.get('GBME_DB_PORT')} "
          f"as {env['GBME_DB_USERNAME']} via {env['GBME_DB_DRIVER']} ...")
    try:
        cn = connect(env)
    except Exception as e:
        print("CONNECT FAILED:", str(e)[:300]); sys.exit(1)
    cur = cn.cursor()
    print("CONNECT OK")

    section("Server")
    try:
        _, v = q(cur, "SELECT @@VERSION"); print(v[0][0].splitlines()[0])
        _, d = q(cur, "SELECT name FROM sys.databases ORDER BY name")
        print("databases:", ", ".join(r[0] for r in d))
    except Exception as e:
        print("version/db list FAILED:", str(e)[:200])

    mf = env.get("GBME_DB_DATABASE_MF", "InsightMart_Test")
    im = env.get("GBME_DB_DATABASE", "InsightMart")
    yardi = env.get("GBME_DB_YARDI", "Yardi7_Reporting")

    section("MULTIFAMILY — stored procs (both candidate DBs)")
    for db in {mf, im}:
        list_procs(cur, db, "sp_Get%")
    print("\nKey multifamily SP presence:")
    for db in {mf, im}:
        for p in ("sp_GetTenantFeaturesByScode", "sp_GetAnchorSnapshot", "sp_GetVacancyByPropertyBedroom"):
            try:
                _, r = q(cur, f"SELECT 1 FROM [{db}].sys.procedures WHERE name='{p}'")
                print(f"    [{db}] {p}: {'FOUND' if r else 'missing'}")
            except Exception as e:
                print(f"    [{db}] {p}: ERR {str(e)[:80]}")

    section("BROKERAGE / MARKETING — InsightMart schemas")
    for sch in ("rea", "mkt", "zi"):
        list_objects(cur, im, sch)
    print("\nKey object row estimates:")
    for (sch, obj) in [("rea", "ProspectScore"), ("rea", "ProspectFeatureSnapshot"),
                       ("rea", "ProspectLabelSnapshot"), ("rea", "ScoreRun"),
                       ("rea", "DealPattern"), ("rea", "EntityConcept"), ("rea", "CompanyInference")]:
        rowcount(cur, im, sch, obj)

    section("YARDI — raw tables (row estimates)")
    for obj in ("TENANT", "PROPERTY", "MM2WO", "MEMO", "CommProspect", "CommAmendment"):
        rowcount(cur, yardi, "dbo", obj)

    cn.close()
    print("\nDONE")


if __name__ == "__main__":
    main()
