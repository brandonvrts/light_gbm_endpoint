"""Read-only data access to the live SQL Server (SSMS source of truth).

Single DB seam for the service. FreeTDS + pyodbc, creds from .env (GBME_DB_*).
Connections are short-lived per call (low request volume in v1). NEVER writes.
"""
from __future__ import annotations
import os
from pathlib import Path

import pandas as pd
import pyodbc

ROOT = Path(__file__).resolve().parent.parent


def _load_env() -> dict:
    env: dict[str, str] = {}
    f = ROOT / ".env"
    if f.exists():
        for line in f.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip("'").strip('"')
    for k, v in os.environ.items():           # OS env overrides .env
        if k.startswith("GBME_"):
            env[k] = v
    return env


ENV = _load_env()
DB_MAIN = ENV.get("GBME_DB_DATABASE", "InsightMart")        # rea.* / mkt.* / zi.*
DB_MF = ENV.get("GBME_DB_DATABASE_MF", "InsightMart_Test")  # multifamily SPs
DB_YARDI = ENV.get("GBME_DB_YARDI", "Yardi7_Reporting")     # raw Yardi tables


def _conn_str() -> str:
    return (
        f"DRIVER={{{ENV['GBME_DB_DRIVER']}}};SERVER={ENV['GBME_DB_SERVER']};"
        f"PORT={ENV.get('GBME_DB_PORT', '1433')};UID={ENV['GBME_DB_USERNAME']};"
        f"PWD={ENV['GBME_DB_PASSWORD']};TDS_Version={ENV.get('GBME_DB_TDS_VERSION', '7.4')};"
    )


def get_connection():
    return pyodbc.connect(_conn_str(), autocommit=True, timeout=15)


def query(sql: str, params: tuple = ()) -> pd.DataFrame:
    """Run a read-only query, return a DataFrame. params bind to ? placeholders."""
    with get_connection() as cn:
        cur = cn.cursor()
        cur.execute(sql, params) if params else cur.execute(sql)
        if cur.description is None:
            return pd.DataFrame()
        cols = [c[0] for c in cur.description]
        rows = [tuple(r) for r in cur.fetchall()]
    return pd.DataFrame.from_records(rows, columns=cols)


def healthcheck() -> dict:
    try:
        df = query("SELECT 1 AS ok")
        return {"db": "ok", "server": ENV.get("GBME_DB_SERVER"), "rows": int(df.iloc[0, 0])}
    except Exception as e:  # noqa: BLE001
        return {"db": "down", "error": str(e)[:200]}
