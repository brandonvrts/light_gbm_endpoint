"""Central config for light_gbm_endpoint.

Adapts the multifamily v2 `config.py` pattern: NO defaults in source; resolve from
`.env` then OS env (GBME_ prefix). Credentials are read here and only consumed by
core/feature_io.py (the single DB seam).

STATUS: scaffold. The loader below is real; wire it into feature_io/auth in Phase 2.
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path

_PREFIX = "GBME_"
HERE = Path(__file__).resolve().parent


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _require(key: str) -> str:
    val = os.environ.get(_PREFIX + key)
    if not val:
        raise RuntimeError(f"Missing required config: {_PREFIX}{key} (set in .env or env)")
    return val


@dataclass(frozen=True)
class DBConfig:
    server: str
    database: str
    username: str
    password: str
    driver: str

    def connection_string(self) -> str:
        return (
            f"DRIVER={{{self.driver}}};SERVER={self.server};DATABASE={self.database};"
            f"UID={self.username};PWD={self.password};TrustServerCertificate=yes;"
        )


@dataclass(frozen=True)
class AADConfig:
    tenant_id: str
    audience: str


def load_db_config(env_file: str | None = None) -> DBConfig:
    _load_dotenv(Path(env_file) if env_file else HERE / ".env")
    return DBConfig(
        server=_require("DB_SERVER"),
        database=_require("DB_DATABASE"),
        username=_require("DB_USERNAME"),
        password=_require("DB_PASSWORD"),
        driver=os.environ.get(_PREFIX + "DB_DRIVER", "ODBC Driver 18 for SQL Server"),
    )


def load_aad_config(env_file: str | None = None) -> AADConfig:
    _load_dotenv(Path(env_file) if env_file else HERE / ".env")
    return AADConfig(tenant_id=_require("AAD_TENANT_ID"), audience=_require("AAD_AUDIENCE"))


def bundles_dir() -> Path:
    return Path(os.environ.get(_PREFIX + "BUNDLES_DIR", str(HERE / "bundles")))
