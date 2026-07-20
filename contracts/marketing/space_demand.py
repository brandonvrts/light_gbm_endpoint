"""Marketing cohort views over the brokerage space-demand model (SAME frozen bundle).

Scores once via signal_desk_spacedemand.score_universe(), tags every scored company with
the canonical marketing cohort key (cohort.py: employee-size band x vertical, industry
from zi.CompanyIndustry PrimaryRank=1), and serves per-cohort rollups + in-cohort lead
lists. Propensities are Platt-calibrated, so a cohort's summed propensity reads as the
expected number of space-need signals in ~180d.

Honest framing (must survive into any UI): the model predicts "will post a relocation/
expansion scoop signal", NOT verified space need and NOT conversion. Only scoop-covered
companies are scorable, so an absent cohort means NO DATA, not no demand.
"""
from __future__ import annotations
import time
import threading
from datetime import date

import numpy as np

from core import db
from contracts.marketing import cohort as cv
from contracts.signal_desk_spacedemand import data as sd

_TTL = 3600
_HOT_QUANTILE = 0.90          # hot = top decile of in-market propensity (audit Phase 0.3 style)
_STATE = {"IA": "Iowa", "NE": "Nebraska"}
_IND_CACHE: dict = {}
_TAG_CACHE: dict = {}
_LOCK = threading.Lock()

_NOTE = ("Cohort view over the PROTOTYPE space-demand model: propensity to post a "
         "relocation/expansion signal (space NEED) in ~180d. NOT a conversion predictor; "
         "only scoop-covered companies are scorable, so missing cohorts mean no data, "
         "not no demand.")


def _primary_industry() -> dict:
    """CompanyID -> primary industry name (zi.CompanyIndustry PrimaryRank=1, which covers
    every company exactly once). TTL-cached."""
    hit = _IND_CACHE.get("m")
    if hit and time.time() - hit[0] < _TTL:
        return hit[1]
    df = db.query(
        f"SELECT ci.CompanyID, i.Name industry FROM [{db.DB_MAIN}].zi.CompanyIndustry ci "
        f"JOIN [{db.DB_MAIN}].zi.Industry i ON i.IndustryID=ci.IndustryID "
        f"WHERE ci.PrimaryRank=1")
    m = {} if df.empty else dict(zip(df["CompanyID"].astype(int), df["industry"]))
    _IND_CACHE["m"] = (time.time(), m)
    return m


def tagged_universe(as_of: str | None = None):
    """The scored universe (score_universe frame) + canonical cohort tags:
    industry, size_band, vertical, cohort_key. Cached per as-of; treat as read-only."""
    asof = (as_of or date.today().isoformat())[:10]
    hit = _TAG_CACHE.get(asof)
    if hit and time.time() - hit[0] < _TTL:
        return hit[1]
    with _LOCK:
        hit = _TAG_CACHE.get(asof)
        if hit and time.time() - hit[0] < _TTL:
            return hit[1]
        df, b = sd.score_universe(asof)
        if not df.empty:
            df = df.copy()
            ind = _primary_industry()
            df["industry"] = df["CompanyID"].map(ind)
            df["size_band"] = df["emp"].map(cv.size_band_of)
            df["vertical"] = df["industry"].map(cv.vertical_of)
            df["cohort_key"] = [
                cv.cohort_key(sb if sb != "unknown" else None, v)
                for sb, v in zip(df["size_band"], df["vertical"])]
        _TAG_CACHE[asof] = (time.time(), (df, b))
        return df, b


def _want_states(market: str | None) -> tuple[list[str], set]:
    mkts = [m.strip() for m in (market.split(",") if market else ["IA", "NE"]) if m.strip()]
    want = set()
    for m in mkts:
        want.add(m); want.add(_STATE.get(m.upper(), m))
    return mkts, want


def _model_source(b) -> dict:
    return {"contract": "signal-desk-spacedemand", "version": b.get("version"),
            "status": "prototype", "label": b.get("label")}


def _lead_row(rank: int, r: dict, pipe: set, hot_thr: float | None) -> dict:
    return {
        "rank": rank, "company_id": r["CompanyID"], "company": r["Name"], "state": r["State"],
        "industry": r.get("industry"), "size_band": r["size_band"],
        "vertical": r.get("vertical"), "cohort_key": r["cohort_key"],
        "propensity": round(float(r["propensity"]), 4),
        "hot": bool(hot_thr is not None and r["propensity"] >= hot_thr),
        "in_pipeline": int(r["CompanyID"]) in pipe,   # already an active RTS prospect
        "signals": {"hiring_365d": int(round(np.expm1(r["hiring_365d"]))),
                    "fundmna_365d": int(r["fundmna_365d"]),
                    "execmove_365d": int(r["execmove_365d"]),
                    "project_365d": int(round(np.expm1(r["project_365d"]))),
                    "had_prior_reloc": bool(r["had_prior_reloc"] >= 1),
                    "days_since_scoop": int(r["days_since_scoop"])},
        "why": sd._why(r)}


def cohort_rollup(market: str | None = None, min_scored: int = 3,
                  as_of: str | None = None) -> dict:
    """Per-cohort demand rollup over the in-market scored universe.

    expected_events_180d = sum of calibrated propensities (expected number of
    space-need signals in the cohort); hot_n = companies in the top decile of the
    in-market propensity distribution (threshold reported so it's auditable)."""
    df, b = tagged_universe(as_of)
    mkts, want = _want_states(market)
    meta = {"view": "space-demand-cohorts", "model_source": _model_source(b),
            "as_of": (as_of or date.today().isoformat())[:10], "market": mkts,
            "hot_definition": f"propensity >= p{int(_HOT_QUANTILE * 100)} of in-market scored set",
            "note": _NOTE}
    if df.empty:
        return {**meta, "totals": {"scored_in_market": 0, "hot_total": 0}, "cohorts": [],
                "verticals": []}
    in_mkt = df[df["State"].isin(want)]
    if in_mkt.empty:
        return {**meta, "totals": {"scored_in_market": 0, "hot_total": 0}, "cohorts": [],
                "verticals": []}
    hot_thr = float(in_mkt["propensity"].quantile(_HOT_QUANTILE))
    pipe = sd._pipeline_company_ids()
    in_pipe = in_mkt["CompanyID"].astype(int).isin(pipe)

    def _stats(g) -> dict:
        hot = g[g["propensity"] >= hot_thr]
        top = g.nlargest(3, "propensity")
        return {
            "n_scored": int(len(g)),
            "expected_events_180d": round(float(g["propensity"].sum()), 1),
            "hot_n": int(len(hot)),
            "hot_new_n": int((~hot["CompanyID"].astype(int).isin(pipe)).sum()),
            "avg_propensity": round(float(g["propensity"].mean()), 4),
            "top_leads": [{"company": t["Name"], "company_id": int(t["CompanyID"]),
                           "propensity": round(float(t["propensity"]), 4)}
                          for t in top.to_dict("records")]}

    cohorts = []
    for key, g in in_mkt.groupby("cohort_key"):
        if len(g) < int(min_scored):
            continue
        sb = g["size_band"].iloc[0]; v = g["vertical"].iloc[0]
        cohorts.append({"cohort_key": key,
                        "size_band": (sb if sb != "unknown" else None),
                        "vertical": (v if isinstance(v, str) else None), **_stats(g)})
    cohorts.sort(key=lambda x: x["expected_events_180d"], reverse=True)

    verticals = []
    for key, g in in_mkt[in_mkt["vertical"].notna()].groupby("vertical"):
        verticals.append({"vertical": key, "cohort_key": cv.cohort_key(None, key), **_stats(g)})
    verticals.sort(key=lambda x: x["expected_events_180d"], reverse=True)

    return {**meta,
            "hot_threshold": round(hot_thr, 4),
            "totals": {"scored_in_market": int(len(in_mkt)),
                       "hot_total": int((in_mkt["propensity"] >= hot_thr).sum()),
                       "expected_events_180d": round(float(in_mkt["propensity"].sum()), 1),
                       "already_in_pipeline": int(in_pipe.sum())},
            "min_scored": int(min_scored),
            "cohorts": cohorts, "verticals": verticals}


def parse_cohort(cohort: str) -> tuple[str | None, str | None]:
    """Parse a cohort selector into (size_band, vertical). Accepts the canonical key
    ('size=mid|vertical=healthcare'), single dims ('size=mid', 'vertical=healthcare'),
    or a bare band/vertical token ('mid', 'healthcare'). Raises ValueError on junk."""
    sb = v = None
    for part in (cohort or "").split("|"):
        part = part.strip()
        if not part or part == "global":
            continue
        if part.startswith("size="):
            sb = part[5:]
        elif part.startswith("vertical="):
            v = part[9:]
        elif part in cv.SIZE_BANDS:
            sb = part
        elif part in cv.VERTICALS:
            v = part
        else:
            raise ValueError(f"unrecognized cohort part '{part}' "
                             f"(size bands: {cv.SIZE_BANDS}; verticals: {cv.VERTICALS})")
    if sb is not None and sb not in cv.SIZE_BANDS:
        raise ValueError(f"unknown size band '{sb}' (one of {cv.SIZE_BANDS})")
    if v is not None and v not in cv.VERTICALS:
        raise ValueError(f"unknown vertical '{v}' (one of {cv.VERTICALS})")
    return sb, v


def cohort_leads(cohort: str, market: str | None = None, limit: int = 25,
                 as_of: str | None = None) -> dict:
    """Ranked in-cohort space-need leads (a filtered view over the same scored frame the
    brokerage endpoint serves — identical propensities by construction)."""
    sb, v = parse_cohort(cohort)
    df, b = tagged_universe(as_of)
    mkts, want = _want_states(market)
    meta = {"view": "space-demand-leads", "model_source": _model_source(b),
            "as_of": (as_of or date.today().isoformat())[:10], "market": mkts,
            "cohort": {"selector": cohort, "size_band": sb, "vertical": v,
                       "cohort_key": cv.cohort_key(sb, v)},
            "note": _NOTE}
    if df.empty:
        return {**meta, "scored_in_cohort": 0, "count": 0, "leads": []}
    in_mkt = df[df["State"].isin(want)]
    hot_thr = float(in_mkt["propensity"].quantile(_HOT_QUANTILE)) if not in_mkt.empty else None
    g = in_mkt
    if sb is not None:
        g = g[g["size_band"] == sb]
    if v is not None:
        g = g[g["vertical"] == v]
    g = g.sort_values("propensity", ascending=False)
    pipe = sd._pipeline_company_ids()
    leads = [_lead_row(i, r, pipe, hot_thr)
             for i, r in enumerate(g.head(int(limit)).to_dict("records"), 1)]
    return {**meta, "hot_threshold": (round(hot_thr, 4) if hot_thr is not None else None),
            "scored_in_cohort": int(len(g)), "count": len(leads),
            "new_vs_pipeline": {"new": sum(1 for x in leads if not x["in_pipeline"]),
                                "already_in_pipeline": sum(1 for x in leads if x["in_pipeline"])},
            "leads": leads}


def cohort_options() -> dict:
    """Non-empty cohort keys over today's scored universe (for UI pickers). Falls back to
    the static vocabulary if scoring/DB is unavailable — options must never 500."""
    try:
        df, b = tagged_universe(None)
        if df.empty:
            raise RuntimeError("empty universe")
        counts = df.groupby("cohort_key").size().sort_values(ascending=False)
        return {"cohort_keys": [{"cohort_key": k, "n_scored": int(n)} for k, n in counts.items()],
                "size_bands": cv.SIZE_BANDS, "verticals": cv.VERTICALS,
                "model_source": _model_source(b)}
    except Exception as e:  # noqa: BLE001
        return {"cohort_keys": [], "size_bands": cv.SIZE_BANDS, "verticals": cv.VERTICALS,
                "error": str(e)[:120]}
