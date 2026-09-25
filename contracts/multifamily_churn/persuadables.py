"""Persuadable residents — the incentive-targeting surface for PMs and the C-suite.

The persuadable band = calibrated churn probability in [0.20, 0.60), no notice filed
(agreed 2026-07-21). These are the residents where an incentive dollar has leverage:
below the band they renew anyway (don't spend / sleeping-dog risk), above it the notice
is filed (mostly lost causes).

Each member carries WHY they're on the fence: their conflicting cohort memberships
(risk vs protective, classified by each cohort's MEASURED live lift, so the labels stay
honest as the population drifts) plus the model's own top drivers for that resident.

Economics (Brandon's parameters): value of a save = avg turn days x the resident's
daily rent + $600 flat make-ready, MINUS the net re-lease premium over one lease term
(we typically re-let turned units above the leaving rent; only the premium above the
renewal increase we'd have taken anyway counts against retention). Turn days and relet
premiums are measured from TENANT consecutive tenancies per property, with fallbacks.
Scenario math takes `incentive_cost` and `flip_rate` as INPUTS — no measured flip rate
exists (TRANS records consumed concessions, not offers), and the Phase 0 review
(docs/PHASE0_DISCOVERY_62W_2026-08-18.md, 2026-08-18) established the historical
observational bound is single digits inside a ±5–13pp confounding floor. Consequently
`flip_rate` no longer defaults: with no rate passed, the scenario reports the
assumption-free BREAKEVEN flip rate (the rate at which spend = value saved) instead of
speculative save counts. The outreach log's incentive_* actions accumulate the
denominator that turns assumption into measurement.

Scoring uses the h180 renewal-forecast bundle explicitly (not the serving default):
the h60 triage ranker compresses nearly the whole population below the band floor
(3 vs 18 band members at 62W — Phase 0 memo §4).
"""
from __future__ import annotations
import time

import pandas as pd

from core import db
from contracts.multifamily_churn import cohorts, contract, data

BAND_LOW, BAND_HIGH = 0.20, 0.60
# The band is defined on the churn probability at the NEXT LEASE DECISION — that is
# the 180-day renewal forecast, not the 60-day triage ranker the registry serves by
# default. Pinned here so a default-bundle change can't silently empty the band.
RENEWAL_BUNDLE = "v6-roadmap-h180"
MAKE_READY_COST = 600.0
FALLBACK_TURN_DAYS = 45.0
_AVG_DAYS_PER_MONTH = 30.44
_RISK_LIFT, _PROTECTIVE_LIFT = 1.05, 0.95

_TURN_CACHE: dict = {}
_RELET_CACHE: dict = {}
_TURN_TTL = 24 * 3600
_EXTENSION_MONTHS = 12          # a save = one more lease term (the 1-yr extension view)


def _turn_days(property_ids: list[int]) -> dict:
    """Avg unit-turn vacancy days per property (move-out -> next move-in on the same
    unit, gaps 0-365d, last 4 years). Cached 24h; portfolio fallback when thin."""
    key = tuple(sorted(property_ids))
    hit = _TURN_CACHE.get(key)
    if hit and time.time() - hit[0] < _TURN_TTL:
        return hit[1]
    ids = ",".join(str(int(p)) for p in property_ids)
    out = {"per_property": {}, "portfolio_avg": FALLBACK_TURN_DAYS,
           "method": "TENANT unit-vacancy gaps (0-365d), last 4 years",
           "fallback_days": FALLBACK_TURN_DAYS}
    try:
        df = db.query(f"""
            SELECT HPROPERTY, AVG(CAST(gap AS FLOAT)) avg_gap, COUNT(*) turns
            FROM (
                SELECT HPROPERTY, DTMOVEOUT,
                       DATEDIFF(DAY, DTMOVEOUT,
                                LEAD(DTMOVEIN) OVER (PARTITION BY HPROPERTY, HUNIT
                                                     ORDER BY DTMOVEIN)) AS gap
                FROM [{db.DB_YARDI}].dbo.TENANT
                WHERE HPROPERTY IN ({ids}) AND DTMOVEIN IS NOT NULL AND {data.OCC_SQL}
            ) g
            WHERE gap >= 0 AND gap <= 365
              AND DTMOVEOUT >= DATEADD(YEAR, -4, GETDATE())
            GROUP BY HPROPERTY""")
        if not df.empty:
            per = {int(r["HPROPERTY"]): {"avg_turn_days": round(float(r["avg_gap"]), 1),
                                         "turns_measured": int(r["turns"])}
                   for r in df.to_dict("records")}
            out["per_property"] = per
            weights = [(v["avg_turn_days"], v["turns_measured"]) for v in per.values()]
            tot = sum(w for _, w in weights)
            if tot:
                out["portfolio_avg"] = round(sum(d * w for d, w in weights) / tot, 1)
    except Exception:  # noqa: BLE001 — economics degrade to the fallback, never 500
        pass
    _TURN_CACHE[key] = (time.time(), out)
    return out


def _relet_premium(property_ids: list[int]) -> dict:
    """Re-lease rent delta per property: next tenant's rent vs the leaving tenant's, on
    the same unit (consecutive tenancies, last 4 years, sane rents, ±50% outliers cut).
    A re-lease PREMIUM shrinks the retention case — losing the resident buys a higher
    market rent — so it's deducted from value-if-saved, NET of the renewal increase we
    would have taken anyway. Cached 24h; failure -> zero adjustment (never blocks)."""
    key = tuple(sorted(property_ids))
    hit = _RELET_CACHE.get(key)
    if hit and time.time() - hit[0] < _TURN_TTL:
        return hit[1]
    ids = ",".join(str(int(p)) for p in property_ids)
    out = {"per_property": {}, "portfolio_avg_pct": 0.0,
           "method": "consecutive-tenancy rent delta on same unit (TENANT, last 4 years)"}
    try:
        df = db.query(f"""
            SELECT HPROPERTY, AVG(delta_pct) avg_pct, COUNT(*) pairs
            FROM (
              SELECT HPROPERTY, DTMOVEOUT,
                     (LEAD(SRENT) OVER (PARTITION BY HPROPERTY, HUNIT ORDER BY DTMOVEIN)
                      - SRENT) / NULLIF(SRENT, 0) * 100.0 AS delta_pct,
                     SRENT,
                     LEAD(SRENT) OVER (PARTITION BY HPROPERTY, HUNIT ORDER BY DTMOVEIN) AS next_rent
              FROM [{db.DB_YARDI}].dbo.TENANT
              WHERE HPROPERTY IN ({ids}) AND DTMOVEIN IS NOT NULL AND {data.OCC_SQL} AND SRENT > 200
            ) g
            WHERE next_rent > 200 AND DTMOVEOUT >= DATEADD(YEAR,-4,GETDATE())
              AND delta_pct BETWEEN -50 AND 50
            GROUP BY HPROPERTY""")
        if not df.empty:
            per = {int(r["HPROPERTY"]): {"relet_premium_pct": round(float(r["avg_pct"]), 2),
                                         "pairs_measured": int(r["pairs"])}
                   for r in df.to_dict("records")}
            out["per_property"] = per
            tot = sum(v["pairs_measured"] for v in per.values())
            if tot:
                out["portfolio_avg_pct"] = round(sum(
                    v["relet_premium_pct"] * v["pairs_measured"] for v in per.values()) / tot, 2)
    except Exception:  # noqa: BLE001
        pass
    _RELET_CACHE[key] = (time.time(), out)
    return out


def _cohort_classes(df: pd.DataFrame) -> dict:
    """Classify every registry cohort by its MEASURED lift on this scored population:
    risk (>=1.05), protective (<=0.95), else neutral."""
    base = float(df["p_churn"].mean())
    classes = {}
    for key, c in cohorts.COHORTS.items():
        g = df[c["predicate"](df)]
        if len(g) == 0 or base == 0:
            classes[key] = {"class": "neutral", "lift": None, "label": c["label"]}
            continue
        lift = float(g["p_churn"].mean() / base)
        cls = "risk" if lift >= _RISK_LIFT else ("protective" if lift <= _PROTECTIVE_LIFT
                                                 else "neutral")
        classes[key] = {"class": cls, "lift": round(lift, 2), "label": c["label"]}
    return classes


def _memberships(row: pd.Series, classes: dict) -> dict:
    """Which cohorts this resident lands in, split risk vs protective (the Venn story)."""
    risk, protect = [], []
    frame = row.to_frame().T
    for key, c in cohorts.COHORTS.items():
        try:
            if not bool(c["predicate"](frame).iloc[0]):
                continue
        except Exception:  # noqa: BLE001
            continue
        cl = classes.get(key, {})
        entry = {"cohort": key, "label": cl.get("label"), "lift": cl.get("lift")}
        if cl.get("class") == "risk":
            risk.append(entry)
        elif cl.get("class") == "protective":
            protect.append(entry)
    return {"risk_cohorts": risk, "protective_cohorts": protect}


def build(property_ids: list[int] | None = None, as_of: str | None = None,
          band_low: float = BAND_LOW, band_high: float = BAND_HIGH,
          incentive_cost: float | None = None, flip_rate: float | None = None,
          top: int = 50, month: str | None = None) -> dict:
    if not (0.0 <= band_low < band_high <= 1.0):
        raise ValueError(f"band must satisfy 0 <= low < high <= 1 (got {band_low}, {band_high})")
    if flip_rate is not None and not (0.0 < flip_rate <= 1.0):
        raise ValueError(f"flip_rate must be in (0, 1] (got {flip_rate})")
    month = (month or "").strip()[:7] or None
    if month:
        try:
            pd.Period(month, freq="M")
            assert len(month) == 7 and month[4] == "-"
        except Exception:
            raise ValueError(f"month must be YYYY-MM (got '{month}')")
    pids = sorted(int(p) for p in (property_ids or data.DEFAULT_PROPERTY_IDS))
    df, b = cohorts._scored_frame(pids, as_of, version=RENEWAL_BUNDLE)
    asof = data._safe_asof(as_of)
    meta = {"contract": "multifamily-churn", "view": "persuadables",
            "model_version": b.get("version"),
            "horizon_days": b.get("horizon_days"), "as_of": asof,
            "properties": [{"property_id": p, "name": data.PROPERTIES.get(p)} for p in pids],
            "month": month,
            "band": {"low": band_low, "high": band_high,
                     "note": "calibrated churn probability at the next lease decision; "
                             "notice-filed residents excluded (lost causes)"}}
    if df.empty:
        return {**meta, "population": {"n": 0}, "n": 0, "members": [],
                "summary": "No active residents found."}

    notice = pd.to_numeric(df.get("HasNotice"), errors="coerce").fillna(0) == 1
    band = df["p_churn"].between(band_low, band_high, inclusive="left") & ~notice
    if month:
        # "edge residents FOR AUGUST" = band members whose lease decision lands in that
        # month (lease_to in YYYY-MM); holdovers only appear in their actual end month
        lease_month = pd.to_datetime(df.get("LeaseToDate"), errors="coerce").dt.strftime("%Y-%m")
        band = band & (lease_month == month)
    g = df[band].copy()
    rent = pd.to_numeric(g.get("CurrentRent"), errors="coerce")
    g["_rent"] = rent.where(rent > 0).fillna(
        pd.to_numeric(g.get("ActiveRent"), errors="coerce")).fillna(0.0)

    base = float(df["p_churn"].mean())
    pop = {"n": int(len(df)), "avg_churn": round(base, 3),
           "expected_moveouts": round(float(df["p_churn"].sum()), 1),
           "notice_given_n": int(notice.sum()),
           "sure_things_n": int(((df["p_churn"] < band_low) & ~notice).sum()),
           "above_band_n": int(((df["p_churn"] >= band_high) & ~notice).sum())}
    if g.empty:
        scope = f" with a lease decision in {month}" if month else ""
        return {**meta, "population": pop, "n": 0, "members": [],
                "summary": f"No residents currently sit in the {band_low}-{band_high} "
                           f"persuadable band without notice{scope}."}

    classes = _cohort_classes(df)
    turns = _turn_days(pids)
    relet = _relet_premium(pids)
    # typical renewal increase per property (what we'd have charged the retained
    # resident anyway) — the relet premium only counts NET of this
    ri = pd.to_numeric(df.get("RentIncreasePct"), errors="coerce")
    pid_col = pd.to_numeric(df.get("PropertyId"), errors="coerce")
    renewal_pct = {int(p): round(float(v), 2)
                   for p, v in ri.groupby(pid_col).mean().dropna().items()}

    # model-side explanations (why/drivers) via the serving path, one batch call
    why_by_scode: dict = {}
    try:
        scored = contract.score_scodes([str(s).strip() for s in g["SCODE"]], asof)
        for r in scored.get("results", []):
            why_by_scode[r.get("scode")] = {"why": r.get("why"), "drivers": r.get("drivers")}
    except Exception:  # noqa: BLE001 — explanations are additive, never fail the list
        pass

    g["_rent_at_risk"] = g["p_churn"] * g["_rent"]
    # per-resident value-if-saved over the WHOLE band (economics must not depend on
    # how many members the caller displays):
    #   turn cost (vacancy + make-ready) − net re-lease premium over one lease term.
    # Re-leasing at a premium means a move-out buys higher market rent, which shrinks
    # what a save is worth — but only the premium ABOVE the renewal increase we would
    # have taken from the retained resident counts.
    def _save_value(r) -> float:
        pid = int(r["PropertyId"]) if pd.notna(r.get("PropertyId")) else None
        t = (turns["per_property"].get(pid) or {}).get("avg_turn_days", turns["portfolio_avg"])
        rent = float(r["_rent"])
        base = t * (rent / _AVG_DAYS_PER_MONTH) + MAKE_READY_COST
        rl = (relet["per_property"].get(pid) or {}).get("relet_premium_pct", relet["portfolio_avg_pct"])
        net_pct = rl - renewal_pct.get(pid, 0.0)
        return round(max(base - net_pct / 100.0 * rent * _EXTENSION_MONTHS, 0.0), 0)
    g["_save_value"] = [_save_value(r) for r in g.to_dict("records")]
    g = g.sort_values("_rent_at_risk", ascending=False)
    shown = g.head(max(int(top), 0))
    # incentive context for the displayed residents (TRANS consumed + logged offers) —
    # best-effort: the edge list must render even if the incentive lookup fails
    try:
        from contracts.multifamily_churn import incentives
        inc_by_scode = incentives.for_scodes([str(s).strip() for s in shown["SCODE"]])
    except Exception:  # noqa: BLE001
        inc_by_scode = {}
    members = []
    for r in shown.to_dict("records"):
        scode = str(r.get("SCODE", "")).strip()
        pid = int(r["PropertyId"]) if pd.notna(r.get("PropertyId")) else None
        mrent = float(r["_rent"])
        exp = why_by_scode.get(scode, {})
        row = pd.Series(r)
        members.append({
            "scode": scode, "full_name": r.get("FullName"),
            "unit": (str(r.get("UnitCode")).strip() if r.get("UnitCode") is not None else None),
            "property": data.PROPERTIES.get(pid) if pid else r.get("PropertyName"),
            "monthly_rent": round(mrent, 2) if mrent else None,
            "lease_to": str(r.get("LeaseToDate") or "")[:10] or None,
            "days_to_lease_end": (int(r["_d2e"]) if pd.notna(r.get("_d2e")) else None),
            "churn_probability": round(float(r["p_churn"]), 4),
            "monthly_rent_at_risk": round(float(r["_rent_at_risk"]), 0),
            "value_if_saved": float(r["_save_value"]),
            "on_the_fence_because": _memberships(row, classes),
            "why": exp.get("why"), "drivers": (exp.get("drivers") or [])[:3],
            "incentives": inc_by_scode.get(scode.lower())})

    # ---- scenario economics (assumptions are INPUTS, labeled as such) ----
    avg_rent = float(g["_rent"].mean())
    cost = float(incentive_cost) if incentive_cost is not None else round(avg_rent, 0)
    exp_moveouts = float(g["p_churn"].sum())
    avg_save_value = float(g["_save_value"].mean())
    spend = len(g) * cost
    # the assumption-free number: the flip rate at which spend = value saved
    breakeven_flip = (round(spend / (exp_moveouts * avg_save_value), 3)
                      if exp_moveouts > 0 and avg_save_value > 0 else None)
    if flip_rate is not None:
        expected_saves = exp_moveouts * flip_rate
        value_saved = expected_saves * avg_save_value
        cost_per_save = round(spend / expected_saves, 0) if expected_saves > 0 else None
    else:
        expected_saves = value_saved = cost_per_save = None
    economics = {
        "turn_days": turns,
        "make_ready_cost": MAKE_READY_COST,
        "relet_premium": {**relet, "renewal_increase_pct_by_property": renewal_pct,
                          "note": "we typically re-lease turned units ABOVE the leaving "
                                  "rent; only the premium net of the renewal increase we'd "
                                  "have taken anyway is deducted from value-if-saved "
                                  f"(over {_EXTENSION_MONTHS} months)"},
        "value_per_save_note": "avg_turn_days x resident daily rent + make-ready flat fee "
                               "- net re-lease premium over one lease term",
        "do_nothing": {
            "expected_moveouts": round(exp_moveouts, 1),
            "turnover_loss": round(float((g["p_churn"] * g["_save_value"]).sum()), 0),
            "note": "the no-intervention baseline: expected move-outs x each resident's "
                    "turn value, this decision window"},
        "scenario": {
            "incentive_cost_per_offer": cost,
            "incentive_cost_source": "input" if incentive_cost is not None
                                     else "default: one month of band avg rent",
            "assumed_flip_rate": flip_rate,
            "flip_rate_note": ("ASSUMED — caller-supplied input, not a measurement"
                               if flip_rate is not None else
                               "NONE PASSED — no measured flip rate exists, and the Phase 0 "
                               "review (2026-08-18) found the historical observational bound "
                               "is single digits inside a ±5–13pp confounding floor; "
                               "breakeven_flip_rate below is the assumption-free read. Log "
                               "offers via POST /multifamily/outreach to measure the real rate"),
            "offers": int(len(g)), "total_spend": round(spend, 0),
            "breakeven_flip_rate": breakeven_flip,
            "breakeven_flip_rate_note": "the flip rate at which spend = value saved; the "
                                        "program pays iff the TRUE rate exceeds this",
            "expected_saves": round(expected_saves, 1) if expected_saves is not None else None,
            "expected_value_saved": round(value_saved, 0) if value_saved is not None else None,
            "net": round(value_saved - spend, 0) if value_saved is not None else None,
            "cost_per_save": cost_per_save,
            "value_per_save": round(avg_save_value, 0),
            "pays": (cost_per_save is not None and cost_per_save <= avg_save_value)
                    if flip_rate is not None else None,
            "breakeven_concession_per_offer": round(
                expected_saves * avg_save_value / len(g), 0)
                if (expected_saves is not None and len(g)) else None,
            "cost_per_save_note": "spend / expected saves — must stay below value_per_save "
                                  "or the program spends a dollar to save less than one"}}

    scope = f" with a lease decision in {month}" if month else ""
    do_nothing = float((g['p_churn'] * g['_save_value']).sum())
    if flip_rate is not None:
        tail = (f"At a ${cost:,.0f} incentive and an ASSUMED {int(flip_rate * 100)}% flip "
                f"rate: {expected_saves:.1f} expected saves worth ${value_saved:,.0f} vs "
                f"${spend:,.0f} spend (net ${value_saved - spend:,.0f}; cost per save "
                f"${(cost_per_save or 0):,.0f} vs ${avg_save_value:,.0f} value).")
    else:
        tail = (f"At a ${cost:,.0f} incentive per offer (${spend:,.0f} total), the program "
                f"pays iff the true flip rate exceeds "
                f"{(breakeven_flip or 0) * 100:.0f}% — no measured rate exists "
                f"(historical bound: single digits), so no save count is projected.")
    summary = (
        f"{len(g)} residents sit in the persuadable band ({band_low}-{band_high} churn "
        f"probability at the {b.get('horizon_days')}-day renewal decision, no notice){scope} "
        f"out of {pop['n']} scored — {exp_moveouts:.1f} expected move-outs worth "
        f"${do_nothing:,.0f} in turns if we do nothing. {tail} "
        f"{pop['sure_things_n']} residents below the band are the do-not-disturb list.")

    return {**meta, "population": pop, "n": int(len(g)),
            "share_pct": round(100 * len(g) / len(df), 1),
            "expected_moveouts": round(exp_moveouts, 1),
            "monthly_rent_at_risk": round(float(g["_rent_at_risk"].sum()), 0),
            "economics": economics, "members": members, "summary": summary}
