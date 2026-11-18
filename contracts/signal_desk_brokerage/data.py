"""Brokerage (Signal Desk) live data access. Pulls a company's full historical
feature snapshots from InsightMart.rea so the model can run through all of them.
"""
from __future__ import annotations
import pandas as pd
from core import db
from contracts.signal_desk_brokerage.schema import FEATURES

_FCOLS = ", ".join(f"f.[{c}]" for c in FEATURES)


def resolve_company(q: str) -> list[dict]:
    """Resolve a company by hMy (numeric) or name -> [{hMy, display_name, market, industry}]."""
    q = (q or "").strip()
    if q.isdigit():
        df = db.query(
            f"SELECT TOP 5 hMy, display_name, market, zi_primary_industry "
            f"FROM [{db.DB_MAIN}].rea.vw_ProspectEnriched WHERE hMy=?", (int(q),))
    else:
        df = db.query(
            f"SELECT TOP 15 hMy, display_name, market, zi_primary_industry "
            f"FROM [{db.DB_MAIN}].rea.vw_ProspectEnriched "
            f"WHERE display_name LIKE ? OR zi_company_name LIKE ? ORDER BY display_name",
            (f"%{q}%", f"%{q}%"))
    return [] if df.empty else df.to_dict("records")


def list_companies(limit: int = 2000) -> list[dict]:
    """ALL prospects that have any feature-snapshot history, alphabetical so the full
    population is selectable (the search box resolves any company by id/name)."""
    df = db.query(
        f"SELECT TOP {int(limit)} e.hMy, e.display_name, e.market, e.zi_primary_industry, "
        f"COUNT(f.SnapshotDate) AS snapshots "
        f"FROM [{db.DB_MAIN}].rea.vw_ProspectEnriched e "
        f"JOIN [{db.DB_MAIN}].rea.ProspectFeatureSnapshot f ON f.hMy=e.hMy "
        f"GROUP BY e.hMy, e.display_name, e.market, e.zi_primary_industry "
        f"HAVING COUNT(f.SnapshotDate) >= 1 ORDER BY e.display_name")
    return [] if df.empty else df.to_dict("records")


def feature_history(hmy: int) -> pd.DataFrame:
    """All snapshots for a company, oldest->newest (the historical-prediction series)."""
    return db.query(
        f"SELECT f.hMy, f.SnapshotDate, {_FCOLS} "
        f"FROM [{db.DB_MAIN}].rea.ProspectFeatureSnapshot f "
        f"WHERE f.hMy=? ORDER BY f.SnapshotDate", (int(hmy),))


_FRESH_CACHE: dict = {}
_FRESH_TTL = 1800
_STALE_AFTER_DAYS = 45   # nightly rea.* job should land within this; else the panel is frozen


def feature_freshness() -> dict:
    """Latest rea.ProspectFeatureSnapshot + rea.ProspectScore dates and a `stale` flag
    (Phase 0.2). The ML feature pipeline froze ~2026-04-30 while rea.CompanyScoreSnapshot
    kept updating, so the conversion endpoint can silently serve months-old inputs; this
    surfaces it. Cached 30 min."""
    import time
    from datetime import date
    hit = _FRESH_CACHE.get("f")
    if hit and time.time() - hit[0] < _FRESH_TTL:
        return hit[1]
    out = {"stale": False, "note": None}
    try:
        f = db.query(f"SELECT MAX(SnapshotDate) d FROM [{db.DB_MAIN}].rea.ProspectFeatureSnapshot")
        s = db.query(f"SELECT MAX(ScoredAt) d FROM [{db.DB_MAIN}].rea.ProspectScore")
        feat_d = f.iloc[0]["d"]
        score_d = s.iloc[0]["d"]
        out["feature_snapshot_latest"] = str(feat_d)[:10] if feat_d is not None else None
        out["rules_score_latest"] = str(score_d)[:10] if score_d is not None else None
        if feat_d is not None:
            fd = feat_d.date() if hasattr(feat_d, "date") else date.fromisoformat(str(feat_d)[:10])
            age = (date.today() - fd).days
            out["feature_snapshot_age_days"] = age
            if age > _STALE_AFTER_DAYS:
                out["stale"] = True
                out["note"] = (f"rea.ProspectFeatureSnapshot last updated {fd} ({age}d ago) — "
                               f"the nightly feature pipeline appears frozen; conversion "
                               f"predictions are point-in-time as of that date, not today.")
    except Exception as e:  # noqa: BLE001
        out["note"] = f"freshness check failed: {str(e)[:120]}"
    _FRESH_CACHE["f"] = (time.time(), out)
    return out


def current_rules_score(hmy: int) -> dict | None:
    df = db.query(
        f"SELECT TOP 1 Probability, Decile, ScoredAt FROM [{db.DB_MAIN}].rea.ProspectScore "
        f"WHERE hMy=? ORDER BY ScoredAt DESC", (int(hmy),))
    return None if df.empty else df.iloc[0].to_dict()


# --- Conversion label (corrected bridge) + hygiene -----------------------------
# The original rea.ProspectLabelSnapshot is broken (resolves to iType=13 "Proposal"
# amendments, not signed leases -> every snapshot of a converting prospect reads Label=1,
# encoding company identity not timing). The corrected label bridges a signed lease via
# vw_Deal.tenantId -> CommAmendmentSnapshot.hTenant, iType in (0,1,2,15), date
# COALESCE(dtSign,dtStart). This function is the SINGLE source of that label (Phase 1.3).
CONV_ITYPES = (0, 1, 2, 15)          # New / Renewal / Expansion / Relocation (signed)
CONV_DATE_MIN = "2015-01-01"         # bridge has garbage dates (pre-2000, post-2030); clamp
CONV_HORIZON_DAYS = 180


def conversion_dates(date_min: str = CONV_DATE_MIN) -> dict[int, list]:
    """hMy -> sorted list of signed-conversion dates (corrected bridge, date-filtered).
    Filtering drops the ~33 nonsensical amendment dates (26 pre-2000, 7 post-2026) that
    would otherwise fabricate labels/censoring."""
    itypes = ",".join(str(int(t)) for t in CONV_ITYPES)
    conv = db.query(
        f"SELECT DISTINCT d.commProspectId AS hMy, COALESCE(a.dtSign, a.dtStart) AS conv_date "
        f"FROM [{db.DB_MAIN}].rea.vw_Deal d "
        f"JOIN [{db.DB_MAIN}].dbo.CommAmendmentSnapshot a ON a.hTenant = d.tenantId "
        f"WHERE a.iType IN ({itypes}) AND COALESCE(a.dtSign, a.dtStart) IS NOT NULL "
        f"  AND COALESCE(a.dtSign, a.dtStart) >= ? "
        f"  AND COALESCE(a.dtSign, a.dtStart) < DATEADD(YEAR, 1, GETDATE())", (date_min,))
    out: dict[int, list] = {}
    if conv.empty:
        return out
    conv["conv_date"] = pd.to_datetime(conv["conv_date"], errors="coerce")
    conv = conv[conv["conv_date"].notna()]
    for r in conv.itertuples(index=False):
        out.setdefault(int(r.hMy), []).append(r.conv_date)
    for h in out:
        out[h].sort()
    return out


def labeled_panel(horizon_days: int = CONV_HORIZON_DAYS,
                  censor_post_conversion: bool = True) -> pd.DataFrame:
    """The hygienic training panel: every ProspectFeatureSnapshot row with a forward-
    looking corrected label (a signed conversion within `horizon_days` AFTER the snapshot).

    censor_post_conversion (Phase 1.3): drop post-first-conversion snapshots that are
    NEGATIVE -- a customer who already signed should not sit in the negative class teaching
    the model "converted => 0". Post-conversion POSITIVES are kept: they are genuine
    renewal/expansion signal (iType 1/2 are in the conversion set). Returns feature cols +
    hMy, SnapshotDate, Label, q (quarter), first_conv_date. Deterministic order."""
    feat = db.query(
        f"SELECT f.hMy, f.SnapshotDate, {_FCOLS} "
        f"FROM [{db.DB_MAIN}].rea.ProspectFeatureSnapshot f ORDER BY f.hMy, f.SnapshotDate")
    if feat.empty:
        return feat
    feat["hMy"] = feat["hMy"].astype("int64")
    feat["SnapshotDate"] = pd.to_datetime(feat["SnapshotDate"], format="mixed", errors="coerce")
    cmap = conversion_dates()
    hi = pd.Timedelta(days=int(horizon_days))

    def _label(hmy, sd):
        ds = cmap.get(int(hmy))
        return int(bool(ds) and any(sd < d <= sd + hi for d in ds))

    feat["Label"] = [_label(h, s) for h, s in zip(feat["hMy"], feat["SnapshotDate"])]
    feat["first_conv_date"] = feat["hMy"].map(lambda h: cmap[h][0] if cmap.get(h) else pd.NaT)
    if censor_post_conversion:
        post = feat["first_conv_date"].notna() & (feat["SnapshotDate"] > feat["first_conv_date"])
        feat = feat[~(post & (feat["Label"] == 0))].reset_index(drop=True)  # drop polluting negatives only
    feat["q"] = feat["SnapshotDate"].dt.year * 4 + (feat["SnapshotDate"].dt.month - 1) // 3
    return feat


def training_data(limit: int | None = None) -> pd.DataFrame:
    """DEPRECATED shim. Historically joined the BROKEN rea.ProspectLabelSnapshot with a
    nondeterministic TOP. Now delegates to the hygienic corrected-label panel so any
    legacy caller (scripts/train_standins.py) gets a correct, deterministic dataset."""
    df = labeled_panel()
    return df.head(int(limit)) if limit else df
