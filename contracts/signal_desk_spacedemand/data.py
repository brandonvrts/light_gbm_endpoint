"""Space-demand (relocation/expansion propensity) live scoring — serves the PROTOTYPE bundle.

Recomputes the SAME 19 point-in-time features as scripts/prototype_spacedemand.py, but for a
SINGLE as-of (default today) across all scooped companies (one row each -> fast, unlike the full
training panel), scores with the frozen v0.1-prototype LightGBM + Platt, and ranks in-market
(IA/NE) 'space-need' leads. TTL-cached: the zi.Scoop panel changes slowly. SELECT-only.
"""
from __future__ import annotations
import time
import threading
from datetime import date

import numpy as np
import pandas as pd

from core import db, registry

RELO = 8
T_HIRING = {11, 12}; T_FUND_MNA = {2, 4}; T_EXECMOVE = {18, 19}
T_PAIN = {20}; T_PROJECT = {21}; T_EARN = {1}; T_AWARD = {6}; T_LAYOFF = {17}
FEATS = ["cnt_90d", "cnt_180d", "cnt_365d", "hiring_365d", "fundmna_365d", "execmove_365d",
         "pain_365d", "project_365d", "earn_365d", "award_365d", "layoff_365d",
         "prior_reloc_ct", "had_prior_reloc", "days_since_scoop", "days_since_growth",
         "momentum_180d", "emp_log", "rev_log", "company_age"]
_STATE = {"IA": "Iowa", "NE": "Nebraska"}
_CACHE: dict = {}
_LOCK = threading.Lock()
_TTL = 3600


def _load():
    s = db.query(
        f"SELECT s.CompanyID, s.OriginalPublishedDate dt, m.TypeID "
        f"FROM [{db.DB_MAIN}].zi.Scoop s LEFT JOIN [{db.DB_MAIN}].zi.ScoopTypeMap m ON m.ScoopID=s.ScoopID "
        f"WHERE s.OriginalPublishedDate IS NOT NULL AND s.CompanyID IS NOT NULL")
    s["dt"] = pd.to_datetime(s["dt"]); s["CompanyID"] = s["CompanyID"].astype("int64")
    s["TypeID"] = pd.to_numeric(s["TypeID"], errors="coerce")
    c = db.query(f"SELECT CompanyID, Name, State, EmployeeCount, Revenue, FoundedYear "
                 f"FROM [{db.DB_MAIN}].zi.Company")
    c["CompanyID"] = c["CompanyID"].astype("int64")
    return s, c


def _features_asof(s, c, asof):
    firmo = {int(r.CompanyID): r for r in c.itertuples(index=False)}
    an = np.datetime64(pd.Timestamp(asof)); D = np.timedelta64(1, "D")
    rows = []
    for cid, g in s.groupby("CompanyID"):
        dts = g["dt"].values.astype("datetime64[ns]"); typ = g["TypeID"].to_numpy()
        prior = dts <= an
        if not prior.any():
            continue
        pdts, ptyp = dts[prior], typ[prior]
        w90 = pdts > an - 90 * D; w180 = pdts > an - 180 * D; w365 = pdts > an - 365 * D
        wprev = (pdts <= an - 180 * D) & (pdts > an - 360 * D)
        gmask = np.isin(ptyp, list(T_HIRING | T_FUND_MNA)); rmask = ptyp == RELO
        fm = firmo.get(int(cid))
        emp = getattr(fm, "EmployeeCount", None) if fm else None
        rev = getattr(fm, "Revenue", None) if fm else None
        fy = getattr(fm, "FoundedYear", None) if fm else None
        rows.append({
            "CompanyID": int(cid), "Name": getattr(fm, "Name", None) if fm else None,
            "State": getattr(fm, "State", None) if fm else None,
            "cnt_90d": float(np.log1p(w90.sum())), "cnt_180d": float(np.log1p(w180.sum())),
            "cnt_365d": float(np.log1p(w365.sum())),
            "hiring_365d": float(np.log1p(np.isin(ptyp[w365], list(T_HIRING)).sum())),
            "fundmna_365d": float(np.isin(ptyp[w365], list(T_FUND_MNA)).sum()),
            "execmove_365d": float(np.isin(ptyp[w365], list(T_EXECMOVE)).sum()),
            "pain_365d": float(np.isin(ptyp[w365], list(T_PAIN)).sum()),
            "project_365d": float(np.log1p(np.isin(ptyp[w365], list(T_PROJECT)).sum())),
            "earn_365d": float(np.isin(ptyp[w365], list(T_EARN)).sum()),
            "award_365d": float(np.isin(ptyp[w365], list(T_AWARD)).sum()),
            "layoff_365d": float(np.isin(ptyp[w365], list(T_LAYOFF)).sum()),
            "prior_reloc_ct": float(rmask.sum()), "had_prior_reloc": float(rmask.any()),
            "days_since_scoop": float((an - pdts.max()) / D),
            "days_since_growth": float((an - pdts[gmask].max()) / D) if gmask.any() else 9999.0,
            "momentum_180d": float(w180.sum() - wprev.sum()),
            "emp_log": float(np.log1p(emp)) if (emp and emp > 0) else np.nan,
            "rev_log": float(np.log1p(rev)) if (rev and rev > 0) else np.nan,
            "company_age": float(pd.Timestamp(asof).year - fy) if (fy and fy > 1800) else np.nan,
        })
    return pd.DataFrame(rows)


def _why(r) -> str:
    bits = []
    if r["had_prior_reloc"] >= 1:
        bits.append("relocated/expanded before")
    if r["days_since_scoop"] <= 30:
        bits.append("very recent activity")
    if r["hiring_365d"] > 0:
        bits.append("recent hiring")
    if r["fundmna_365d"] > 0:
        bits.append("funding/M&A")
    if r["execmove_365d"] > 0:
        bits.append("exec change")
    if r["project_365d"] > 0:
        bits.append("new project")
    return ", ".join(bits[:3]) if bits else "firmographic profile"


def _metrics(bundle) -> dict:
    m = (bundle.get("metrics") or {}).get("lightgbm") or {}
    base = m.get("base_rate") or 0.0
    return {"pr_auc": m.get("pr_auc"), "roc_auc": m.get("roc_auc"), "base_rate": base,
            "pr_auc_vs_base_x": round((m.get("pr_auc") or 0) / base, 1) if base else None,
            "pr_auc_lift_ci95": m.get("pr_auc_lift_ci95"),
            "positive_company_months": m.get("oof_pos")}


def leads(market: str | None = None, limit: int = 25, as_of: str | None = None) -> dict:
    asof = (as_of or date.today().isoformat())[:10]
    mkts = [m.strip() for m in (market.split(",") if market else ["IA", "NE"]) if m.strip()]
    want = set()
    for m in mkts:
        want.add(m); want.add(_STATE.get(m.upper(), m))
    key = (asof, tuple(sorted(want)), int(limit))
    hit = _CACHE.get(key)
    if hit and time.time() - hit[0] < _TTL:
        return hit[1]
    with _LOCK:
        hit = _CACHE.get(key)
        if hit and time.time() - hit[0] < _TTL:
            return hit[1]
        b = registry.load("signal_desk_spacedemand")
        s, c = _load()
        df = _features_asof(s, c, asof)
        meta = {"model_version": b.get("version"), "status": "prototype", "as_of": asof,
                "label": b.get("label"), "market": mkts, "metrics": _metrics(b),
                "note": ("PROTOTYPE lead-gen: ranks in-market companies by propensity to post a "
                         "relocation/expansion signal (space NEED) in ~180d. NOT a conversion "
                         "predictor and NOT broker-facing truth — a research prioritization list.")}
        if df.empty:
            out = {**meta, "count": 0, "leads": []}
            _CACHE[key] = (time.time(), out); return out
        raw = b["model"].predict_proba(df[FEATS])[:, 1]
        platt = b.get("platt")
        df["propensity"] = platt.predict_proba(raw.reshape(-1, 1))[:, 1] if platt is not None else raw
        m = df[df["State"].isin(want)].sort_values("propensity", ascending=False).head(int(limit))
        out_leads = []
        for i, r in enumerate(m.to_dict("records"), 1):
            out_leads.append({
                "rank": i, "company_id": r["CompanyID"], "company": r["Name"], "state": r["State"],
                "propensity": round(float(r["propensity"]), 4),
                "signals": {"hiring_365d": int(round(np.expm1(r["hiring_365d"]))),
                            "fundmna_365d": int(r["fundmna_365d"]),
                            "execmove_365d": int(r["execmove_365d"]),
                            "project_365d": int(round(np.expm1(r["project_365d"]))),
                            "had_prior_reloc": bool(r["had_prior_reloc"] >= 1),
                            "days_since_scoop": int(r["days_since_scoop"])},
                "why": _why(r)})
        out = {**meta, "scored_in_market": int(len(df[df["State"].isin(want)])), "count": len(out_leads),
               "leads": out_leads}
        _CACHE[key] = (time.time(), out)
        return out
