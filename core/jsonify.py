"""Coerce DB/numpy types to JSON-safe Python (Decimal->float, dates->iso, NaN->None).
Addresses the Phase 0 eval flags (pyodbc Decimal, datetimes, NULLs)."""
from __future__ import annotations
from datetime import date, datetime
from decimal import Decimal
import math
import numpy as np


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, Decimal):
        return float(o)
    if isinstance(o, (datetime, date)):
        return o.isoformat()
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        f = float(o)
        return None if math.isnan(f) else f
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, float) and math.isnan(o):
        return None
    if isinstance(o, str):
        return o.strip()
    return o
