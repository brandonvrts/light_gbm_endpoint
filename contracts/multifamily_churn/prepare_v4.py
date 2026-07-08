"""Feature prep for the REAL production v4 bundle (scaler + sklearn LabelEncoders +
feature_medians). Faithful port of v2 validate_april_renewals_v3.prepare_features so
inference matches training exactly. Used when a non-stand-in bundle is loaded.
"""
from __future__ import annotations
import pandas as pd

_ENC = {"PropertyName_encoded", "PrimaryPetType_encoded"}


def prepare(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    prop_enc = bundle["property_encoder"]
    pet_enc = bundle["pet_encoder"]
    fcols = bundle["feature_columns"]
    med = bundle.get("feature_medians") or {}
    df = df.copy()

    known_props = set(prop_enc.classes_)
    prop_fb = prop_enc.classes_[0]
    df["PropertyName"] = (df.get("PropertyName").fillna(prop_fb).astype(str)
                          .apply(lambda x: x if x in known_props else prop_fb))
    df["PropertyName_encoded"] = prop_enc.transform(df["PropertyName"])

    known_pets = set(pet_enc.classes_)
    pet_fb = "None" if "None" in known_pets else pet_enc.classes_[0]
    df["PrimaryPetType"] = (df.get("PrimaryPetType").fillna(pet_fb).astype(str)
                            .apply(lambda x: x if x in known_pets else pet_fb))
    df["PrimaryPetType_encoded"] = pet_enc.transform(df["PrimaryPetType"])

    for col in fcols:
        if col in _ENC:
            continue
        s = pd.to_numeric(df.get(col), errors="coerce")
        fill = med.get(col)
        if fill is None:
            fill = s.median() if s.notna().any() else 0.0
        df[col] = s.fillna(fill)

    return df[fcols].astype(float)
