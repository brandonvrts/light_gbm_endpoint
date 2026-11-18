"""Adapter so the modules ported from multifamily-prediction `v2/` run unchanged here.

The branch modules did `from config import DATA_DIR, MODEL_DIR, get_connection`; this
supplies the same three names backed by this repo's read-only DB seam (core.db).
Training artifacts (CSVs, walk-forward pools) go under data/renewal_v8/ -- they hold
resident-level rows and stay out of git.
"""
from __future__ import annotations

from core import db

DATA_DIR = str(db.ROOT / "data" / "renewal_v8")
MODEL_DIR = DATA_DIR   # walk-forward JSON lands next to its pool; bundles go to bundles/


def get_connection():
    return db.get_connection()
