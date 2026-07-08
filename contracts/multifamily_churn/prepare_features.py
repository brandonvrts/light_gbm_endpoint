"""Multifamily feature prep — shared by training and inference (no train/inference skew).

Raw sp_GetTenantFeaturesByScode rows -> 45-feature float matrix. Encoders + medians
are fit at train time and stored in the bundle; inference reuses them.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from contracts.multifamily_churn.data import MODEL_FEATURES

NUMERIC = [f for f in MODEL_FEATURES if not f.endswith("_encoded")]  # 43


def fit_prep(df: pd.DataFrame) -> tuple[dict, dict]:
    medians = {}
    for c in NUMERIC:
        m = pd.to_numeric(df.get(c), errors="coerce").median() if c in df else np.nan
        medians[c] = 0.0 if pd.isna(m) else float(m)
    pvals = sorted({str(v) for v in df.get("PropertyName", pd.Series(dtype=object)).dropna()})
    petvals = sorted({("None" if pd.isna(v) else str(v))
                      for v in df.get("PrimaryPetType", pd.Series(dtype=object))})
    encoders = {"PropertyName": {v: i + 1 for i, v in enumerate(pvals)},
                "PrimaryPetType": {v: i + 1 for i, v in enumerate(petvals)}}
    return encoders, medians


def prepare(df: pd.DataFrame, encoders: dict, medians: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in NUMERIC:
        s = pd.to_numeric(df.get(c), errors="coerce")
        out[c] = s.fillna(medians.get(c, 0.0)).astype(float)
    pmap = encoders.get("PropertyName", {})
    petmap = encoders.get("PrimaryPetType", {})
    prop = df["PropertyName"] if "PropertyName" in df else pd.Series([None] * len(df))
    pet = df["PrimaryPetType"] if "PrimaryPetType" in df else pd.Series([None] * len(df))
    out["PropertyName_encoded"] = [pmap.get(str(v), 0) for v in prop]
    out["PrimaryPetType_encoded"] = [petmap.get("None" if pd.isna(v) else str(v), 0) for v in pet]
    return out[MODEL_FEATURES]
