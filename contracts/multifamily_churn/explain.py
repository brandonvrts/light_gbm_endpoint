"""Per-resident score explanations ("why") from the bundle's own LightGBM.

SHAP-style pred_contrib from the underlying booster, translated into short
human-readable clauses a chatbot can read back verbatim. Motivation: without this the
chatbot invented its own (wrong) explanation for a score -- it guessed tenure when the
real driver was maintenance activity. Never raises: any failure returns empty
explanations rather than breaking scoring.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# feature -> (phrase when pushing risk UP, phrase when pulling risk DOWN).
# {v} = the resident's raw feature value. None = don't mention in that direction.
_PHRASES: dict[str, tuple[str | None, str | None]] = {
    "HasNotice": ("a move-out notice is on file", "no move-out notice on file"),
    "DaysSinceNotice": ("notice given {v:.0f} days ago", None),
    "DaysToLeaseEnd": ("the lease ends in {v:.0f} days", "the lease still has {v:.0f} days to run"),
    "IsLeaseExpiringSoon": ("the lease expires within 90 days", "the lease is not near expiration"),
    "LeaseTermMonths": ("a {v:.0f}-month lease term", "a {v:.0f}-month lease term"),
    "PlannedLeaseDays": ("a {v:.0f}-day planned lease", None),
    "TenureMonths": ("short tenure ({v:.0f} months)", "long tenure ({v:.0f} months / {yrs:.0f} years)"),
    "TenureDaysAtAnchor": ("short tenure ({yrs:.1f} years)", "long tenure ({yrs:.0f} years)"),
    "MoveInYear": (None, None),        # redundant with tenure -- skip in prose
    "MoveInMonth": (None, None),
    "DecisionMonth": (None, None),     # calendar seasonality -- meaningless in prose
    "DecisionYear": (None, None),
    "DecisionNumber": ("first renewal decision", "{v:.0f} prior renewal decisions"),
    "CurrentRent": ("rent level (${v:,.0f})", "rent level (${v:,.0f})"),
    "ActiveRent": ("rent level (${v:,.0f})", "rent level (${v:,.0f})"),
    "PriorRent": (None, None),
    "RentIncreasePct": ("a {v:.1f}% rent increase", "no meaningful rent increase"),
    "NSFCount": ("{v:.0f} returned payments", None),
    "NoPaymentsFlag": ("no payments on record", None),
    "WorkOrderCount": ("{v:.0f} work orders in the last 18 months", "little maintenance activity"),
    "WorkOrderFrequency": ("frequent work orders", "infrequent work orders"),
    "EmergencyWorkOrders": ("{v:.0f} emergency work orders", None),
    "EmergencyWO_Last90Days": ("{v:.0f} emergency work orders in the last 90 days", None),
    "WorkOrders_Last90Days": ("{v:.0f} work orders in the last 90 days", None),
    "WorkOrders_Last180Days": ("{v:.0f} work orders in the last 180 days", None),
    "WorkOrders_Last365Days": ("{v:.0f} work orders in the last year", None),
    "DaysSinceLastWorkOrder": ("a work order {v:.0f} days ago", "no recent work orders"),
    "OpenWorkOrdersAtAnchor": ("{v:.0f} still-open work order(s)", "no open work orders"),
    "MaxOpenWorkOrderAge": ("an open work order aging {v:.0f} days", None),
    "EarlyWorkOrders30d": ("work orders within 30 days of moving in", None),
    "AvgDaysToComplete": ("slow work-order resolution ({v:.0f}-day average)",
                          "fast work-order resolution"),
    "MemoCount": ("{v:.0f} account memos", "a quiet account file"),
    "MemoFrequency": ("frequent account memos", None),
    "Memos_Last90Days": ("{v:.0f} memos in the last 90 days", None),
    "Memos_Last180Days": (None, None),
    "Memos_Last365Days": (None, None),
    "DaysSinceLastMemo": (None, None),
    "EvictionMemos": ("eviction-related memos", None),
    "PaymentMemos": ("payment-related memos", None),
    "ComplaintMemos": ("complaint memos on file", None),
    "MaintenanceMemos": ("maintenance-complaint memos", None),
    "PetCount": (None, None),
    "HasLargePets": (None, None),
    "HasLeaseRecord": (None, "an on-file lease record"),
    "PropertyName_encoded": ("the property's churn base rate ({prop})",
                             "the property's churn base rate ({prop})"),
    "PrimaryPetType_encoded": (None, None),
}


def _booster(bundle: dict):
    """Unwrap PlattRescaler -> CalibratedClassifierCV -> FrozenEstimator -> LGBM booster."""
    m = bundle.get("model")
    m = getattr(m, "inner_model", m)
    if hasattr(m, "calibrated_classifiers_"):
        inner = m.calibrated_classifiers_[0]
        inner = getattr(inner, "estimator", getattr(inner, "base_estimator", inner))
        m = getattr(inner, "estimator", inner)
    return getattr(m, "booster_", None)


def contributions(bundle: dict, X_scaled) -> np.ndarray | None:
    """Per-row, per-feature log-odds contributions (last column = bias), or None."""
    try:
        bst = _booster(bundle)
        if bst is None:
            return None
        return np.asarray(bst.predict(X_scaled, pred_contrib=True))
    except Exception:
        return None


def _clause(feat: str, value, up: bool, prop_name: str | None) -> str | None:
    tpl = _PHRASES.get(feat, (f"{feat} ({{v}})", f"{feat} ({{v}})"))[0 if up else 1]
    if tpl is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = 0.0
    return tpl.format(v=v, yrs=v / (12.0 if "Months" in feat else 365.25),
                      prop=prop_name or "this property")


def why(bundle: dict, X: pd.DataFrame, contrib: np.ndarray, i: int,
        top_k: int = 3, prop_name: str | None = None) -> dict:
    """One resident's explanation: short `why` sentence + structured `drivers` list."""
    cols = list(X.columns)
    c = contrib[i][:len(cols)]
    order = np.argsort(-np.abs(c))
    ups, downs, drivers = [], [], []
    for j in order:
        if len(ups) >= top_k and len(downs) >= 2:
            break
        feat, w = cols[j], float(c[j])
        if abs(w) < 0.05:               # negligible -- stop once signal fades
            break
        clause = _clause(feat, X.iloc[i][feat], w > 0, prop_name)
        if clause is None:
            continue
        drivers.append({"feature": feat, "direction": "up" if w > 0 else "down",
                        "weight": round(w, 3), "value": round(float(X.iloc[i][feat]), 2)})
        if w > 0 and len(ups) < top_k and clause not in ups:
            ups.append(clause)
        elif w < 0 and len(downs) < 2 and clause not in downs:
            downs.append(clause)
    txt = ""
    if ups:
        txt = "Driven by " + ", ".join(ups)
        if downs:
            txt += "; offset by " + " and ".join(downs)
    elif downs:
        txt = "Low risk mainly because of " + " and ".join(downs)
    return {"why": txt + "." if txt else None, "drivers": drivers}
