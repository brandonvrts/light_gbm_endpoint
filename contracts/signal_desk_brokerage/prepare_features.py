"""Brokerage feature prep — shared by training and inference. All 37 features are
numeric/bit; just coerce + impute medians (stored in the bundle)."""
from __future__ import annotations
import numpy as np
import pandas as pd

from contracts.signal_desk_brokerage.schema import FEATURES


def fit_prep(df: pd.DataFrame) -> dict:
    medians = {}
    for c in FEATURES:
        m = pd.to_numeric(df.get(c), errors="coerce").median() if c in df else np.nan
        medians[c] = 0.0 if pd.isna(m) else float(m)
    return medians


def prepare(df: pd.DataFrame, medians: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in FEATURES:
        s = pd.to_numeric(df.get(c), errors="coerce")
        out[c] = s.fillna(medians.get(c, 0.0)).astype(float)
    return out[FEATURES]
