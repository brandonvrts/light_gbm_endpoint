"""Serve the v8 renewal-decision bundle (feature_semantics "v8-renewal-pit").

Mirrors multifamily-prediction v2/score_forward.py (branch commit 195cceb) with one
change: score_forward refits LightGBM on every run, here the model is fit once by
scripts/train_multifamily_v8.py and frozen in the bundle (trained on every lease event
whose outcome had resolved at train time -- the same rule score_forward applies at
its anchor).

Per resident:
  features  = pit_features.fetch_pit_features (anchor = as_of) + extra_features
              (tenant_history notice, ledger NSF, ...) evaluated at as_of
  lease end = earliest Lease_History dtLeaseTo >= as_of with sStatus Current/Past.
              DaysToLeaseEnd = lease end - as_of.
              NOT 'Approved' -- deliberate deviation from the branch's score_forward.py.
              Verified 2026-11-12: Approved rows are the UNCHOSEN renewal term options
              (12/15/18-mo offers bulk-approved together; the one the resident picks
              becomes Current, the rest stay Approved forever). e.g. t0003401: Approved
              ->2027-01-05 and ->2027-04-05, Current ->2027-07-05 = TENANT.DTLEASETO.
              Including them put 31 phantom residents in the Jan-2027 cohort.
  p_raw     = model probability (use for RANKING / outreach queues)
  p         = isotonic-calibrated p_raw, clipped (use for STATED probabilities)

Residents whose lease end is outside the trained 45-210 day lead window, or whose
property was not in training, are still scored but flagged.
"""
from __future__ import annotations

import threading
import time
from datetime import date

import numpy as np
import pandas as pd

from core import db
from . import extra_features as XF
from . import pit_features as PF

# readable driver names (branch score_mom_retrained.DRIVER_LABELS, abridged)
DRIVER_LABELS = {
    'TenureMonths': 'tenure', 'TenureDaysAtAnchor': 'tenure',
    'DaysToLeaseEnd': 'days to lease end', 'PlannedLeaseDays': 'lease length',
    'LeaseTermMonths': 'lease term', 'DecisionNumber': 'renewal number',
    'CurrentRent': 'current rent', 'ActiveRent': 'current rent',
    'PriorRent': 'prior rent', 'RentIncreasePct': 'rent increase %',
    'MoveInMonth': 'move-in month', 'MoveInYear': 'move-in year',
    'DecisionMonth': 'month of decision', 'DecisionYear': 'year of decision',
    'IsLeaseExpiringSoon': 'lease expiring soon',
    'WorkOrderCount': 'work orders (18 mo)', 'WorkOrderFrequency': 'work-order rate',
    'EmergencyWorkOrders': 'emergency work orders',
    'AvgDaysToComplete': 'work-order completion time',
    'WorkOrders_Last90Days': 'work orders (90d)', 'WorkOrders_Last180Days': 'work orders (180d)',
    'WorkOrders_Last365Days': 'work orders (365d)',
    'EmergencyWO_Last90Days': 'emergency work orders (90d)',
    'DaysSinceLastWorkOrder': 'days since last work order',
    'OpenWorkOrdersAtAnchor': 'open work orders', 'MaxOpenWorkOrderAge': 'oldest open work order',
    'EarlyWorkOrders30d': 'work orders in first 30 days',
    'MemoCount': 'memos (18 mo)', 'MemoFrequency': 'memo rate',
    'Memos_Last90Days': 'memos (90d)', 'Memos_Last180Days': 'memos (180d)',
    'Memos_Last365Days': 'memos (365d)', 'DaysSinceLastMemo': 'days since last memo',
    'EvictionMemos': 'eviction memos', 'PaymentMemos': 'payment memos',
    'ComplaintMemos': 'complaint memos', 'MaintenanceMemos': 'maintenance memos',
    'PetCount': 'pets', 'HasLargePets': 'large pet',
    'PropertyName_encoded': 'property', 'PrimaryPetType_encoded': 'pet type',
    'X_ev_notice_active': 'notice on file', 'X_ev_days_since_notice': 'days since notice',
    'X_pay_nsf_cnt': 'NSF payments (12 mo)',
}


def _lease_end_sql(n: int) -> str:
    Y = f"[{db.DB_YARDI}].dbo"
    marks = ",".join("?" * n)
    return f"""
    SELECT LOWER(RTRIM(t.SCODE)) AS scode, MIN(lh.dtLeaseTo) AS LeaseEnd
    FROM {Y}.Lease_History lh
    JOIN {Y}.TENANT t ON t.HMYPERSON = lh.hTent
    WHERE lh.dtLeaseTo >= ? AND lh.sStatus IN ('Current', 'Past')
      AND LOWER(RTRIM(t.SCODE)) IN ({marks})
    GROUP BY LOWER(RTRIM(t.SCODE))"""


def upcoming_lease_end(scodes: list[str], as_of: str, batch: int = 500) -> pd.DataFrame:
    clean = sorted({s.strip().lower() for s in scodes if s and s.strip()})
    parts = []
    for i in range(0, len(clean), batch):
        chunk = clean[i:i + batch]
        d = db.query(_lease_end_sql(len(chunk)), tuple([as_of] + chunk))
        if not d.empty:
            parts.append(d)
    if not parts:
        return pd.DataFrame(columns=["scode", "LeaseEnd"])
    out = pd.concat(parts, ignore_index=True)
    out["LeaseEnd"] = pd.to_datetime(out["LeaseEnd"], errors="coerce")
    return out


# ---- daily cache of the extra-feature bulk pulls ------------------------------------
# XF.load_raw() pulls whole-portfolio tables (~1M ledger rows, ~10 s) that are the
# same for every resident. Cache one copy per calendar day. The PIT cut at as_of happens
# downstream, so the cache is valid for any anchor. Staleness: at most since the day's
# first pull (a notice logged in Yardi after that shows up the next day, or call
# refresh_raw_cache()). After midnight the first request is served from yesterday's
# pull while a background thread refreshes, so only a cold start blocks.
_RAW: dict = {"day": None, "raw": None, "pulled_at": None}
_RAW_LOCK = threading.Lock()
_REFRESHING = threading.Event()


def _pull_into_cache() -> dict:
    raw = XF.load_raw()
    _RAW.update(day=date.today(), raw=raw, pulled_at=time.time())
    return raw


def _bg_refresh() -> None:
    try:
        with _RAW_LOCK:
            if _RAW["day"] != date.today():
                _pull_into_cache()
    except Exception:      # keep serving the previous day's pull; retry next request
        pass
    finally:
        _REFRESHING.clear()


def cached_raw() -> dict:
    if _RAW["raw"] is not None:
        if _RAW["day"] != date.today() and not _REFRESHING.is_set():
            _REFRESHING.set()
            threading.Thread(target=_bg_refresh, daemon=True).start()
        return _RAW["raw"]
    with _RAW_LOCK:              # cold start: block once, concurrent callers wait
        if _RAW["raw"] is None:
            _pull_into_cache()
        return _RAW["raw"]


def refresh_raw_cache() -> None:
    """Force a fresh pull (e.g. after a known Yardi update)."""
    with _RAW_LOCK:
        _pull_into_cache()


def raw_cache_info() -> dict:
    return {"day": str(_RAW["day"]) if _RAW["day"] else None,
            "age_s": round(time.time() - _RAW["pulled_at"]) if _RAW["pulled_at"] else None}


def features(scodes: list[str], as_of: str, batch: int = 400) -> pd.DataFrame:
    """Branch PIT feature block + extras + upcoming lease end, one row per resident."""
    clean = [s.strip() for s in scodes if s and s.strip()]
    if not clean:
        return pd.DataFrame()
    frames = [PF.fetch_pit_features(clean[i:i + batch], as_of)
              for i in range(0, len(clean), batch)]
    f = pd.concat([x for x in frames if not x.empty], ignore_index=True) if any(
        not x.empty for x in frames) else pd.DataFrame()
    if f.empty:
        return f
    f = f.rename(columns={"HMYPERSON": "hMyPerson"})
    f["hMyPerson"] = pd.to_numeric(f["hMyPerson"]).astype("int64")
    f["scode"] = f["SCODE"].astype(str).str.strip().str.lower()
    f["AnchorDate"] = as_of
    extras = XF.build_extra_features(f[["hMyPerson", "AnchorDate"]], raw=cached_raw())
    f = f.merge(extras, on=["hMyPerson", "AnchorDate"], how="left")
    le = upcoming_lease_end(f["scode"].tolist(), as_of)
    f = f.merge(le, on="scode", how="left")
    # forward run: the authoritative upcoming expiration drives DaysToLeaseEnd
    f["AnchorLeaseToDate"] = f["LeaseEnd"]
    f["DaysToLeaseEnd"] = (f["LeaseEnd"] - pd.Timestamp(as_of)).dt.days
    # DISPLAY / cohort lease end: holdover & month-to-month residents have no upcoming
    # term, so fall back to their (expired) TENANT lease end -> negative days, which the
    # holdover cohort keys off. The model input above stays NaN (median) for them --
    # they are outside the trained population and get flagged.
    f["LeaseToDisplay"] = f["LeaseEnd"].fillna(pd.to_datetime(f.get("LeaseToDate"), errors="coerce"))
    # preserve request order
    order = {s.lower(): k for k, s in enumerate(clean)}
    f["_o"] = f["scode"].map(order)
    return f.sort_values("_o").drop(columns="_o").reset_index(drop=True)


def _encode(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    d = df.copy()
    for raw, enc in (("PropertyName", "PropertyName_encoded"),
                     ("PrimaryPetType", "PrimaryPetType_encoded")):
        spec = bundle["encoders"][raw]
        classes = list(spec["classes"])
        fallback = spec["fallback"]
        v = d[raw].fillna(fallback).astype(str) if raw in d else pd.Series(fallback, index=d.index)
        v = v.where(v.isin(classes), fallback)
        d[enc] = v.map({c: i for i, c in enumerate(classes)})
    return d


def calibrate(p_raw, bundle: dict) -> np.ndarray:
    cal = bundle["calibration"]
    lo, hi = cal["clip"]
    return np.clip(np.interp(np.asarray(p_raw, float), cal["grid_x"], cal["grid_y"]), lo, hi)


def predict(df: pd.DataFrame, bundle: dict):
    """-> (p_calibrated, p_raw, contribs[n, k], X) for rows of `features()` output."""
    d = _encode(df, bundle)
    cols = bundle["feature_columns"]
    med = pd.Series(bundle["feature_medians"])
    X = d.reindex(columns=cols).apply(pd.to_numeric, errors="coerce").fillna(med).fillna(0)
    m = bundle["model"]
    p_raw = m.predict_proba(X)[:, 1]
    contrib = m.booster_.predict(X, pred_contrib=True)[:, :-1]
    return calibrate(p_raw, bundle), p_raw, contrib, X


def drivers(cols: list[str], contrib_row: np.ndarray, k: int = 3) -> list[dict]:
    order = np.argsort(-np.abs(contrib_row))[:k]
    return [{"feature": cols[i], "label": DRIVER_LABELS.get(cols[i], cols[i]),
             "direction": "raises" if contrib_row[i] > 0 else "lowers",
             "log_odds": round(float(contrib_row[i]), 4)}
            for i in order if abs(contrib_row[i]) >= 1e-6]
