"""Concession cohort — the economics intern's modeling surface (2026-09-25).

The question this view answers: "which residents does the model say are at real risk
of leaving at their next lease decision, and what would each concession cost against
what losing them costs?" Every input is a resident-level, assumption-free number so the
intern can model the three offer families (free garage / one month free / 5% off)
against NOI herself. Nothing here assumes an uplift unless the caller passes one.

Cohort definition (defaults; every knob is a query parameter):
  * churn probability at the NEXT lease decision in [0.20, 0.60) — the persuadable band
    agreed 2026-07-21 and shared with /persuadables and the edge roster — scored on the
    h180 renewal-forecast bundle. The requested "30–40% chance of staying" band
    ([0.60, 0.70) churn) is EMPTY on this model (2026-09-25: no-notice p99 = 0.39, one
    resident above 0.5): notice is the dominant feature, so without one the model never
    gets that confident. The h60 triage ranker the registry serves by default compresses
    nearly everyone below 0.20 (Phase 0 memo §4), so h180 is pinned here exactly as
    persuadables.py does. Calibrated, UNclipped. ~150 residents today; band_low=0.30
    keeps the top ~35.
  * lease decision within 180 days (real calendar LeaseToDate − as_of). Holdovers
    (negative distance) are overdue decisions and stay in.
  * no move-out notice filed. Notice is ~terminal (2.7% renew after it) and is a
    different population with its own economics; include_notice=true to see them.

Rows are identifier-only by design — scode / hmyperson / unit / property, no names —
and carry: lease timing, rent (in-place, prior, market), tenure, the h180 probability
with the bundle's OWN drivers (not the h60 serving path's), risk/protective cohort
memberships, garage status from RENTABLEITEMS (stalls held + rent), consumed
concessions + logged offers, the INTERNAL duration-model tenure estimates, the full
model feature vector, and the per-bedroom turnover economics from the Phase 1 ledger.

Concession economics per resident (docs/INTERN_RETENTION_ECONOMICS_HANDOFF.md §3–§4):
  E[net gain | offer] = Δ·TC_net − (1 − p + Δ)·C        C is paid only if they renew
  breakeven Δ*        = (1 − p)·C / (TC_net − C)          "never" if C ≥ TC_net
  max spend C_max     = Δ·TC_net / (1 − p + Δ)
Offer cost models: one month free = rent (one-time credit); 5% off = 0.6 months of rent
term-limited, with the baked-into-base 3-year cost shown alongside; free garage splits
into a NEW assignment ((12 − E[months to re-rent]) × the vacant stall's rent, only when
the resident has no garage and a stall is vacant) and a WAIVER of an existing stall
(12 × what they pay now). `verdict` buckets the breakeven against what history can
support: green ≤ 5pp, amber ≤ 15pp, red above, never when C ≥ TC_net.
"""
from __future__ import annotations

import csv
import io
import json
import threading
import time

import numpy as np
import pandas as pd

from core import db
from contracts.multifamily_churn import (cohorts, contract, data, explain, persuadables,
                                         prepare_v4, unit_economics)

BAND_LOW, BAND_HIGH = 0.20, 0.60
DECISION_WINDOW_DAYS = 180
RENEWAL_BUNDLE = persuadables.RENEWAL_BUNDLE          # v6-roadmap-h180
GARAGE_CHARGE_CODE = 17           # RENTABLEITEMSTYPE.HCHARGECODE for garage/parking types
GARAGE_MONTHS_TO_RERENT = 2.0     # Phase 0 median at 62W — the vacant stall's opportunity horizon
DISCOUNT_PCT = 5.0
DISCOUNT_BASE_YEARS = 3
DEFAULT_RENT_GROWTH_PCT = 3.0     # baked-into-base compounding when no renewal increase is measured
VERDICT_GREEN_PP, VERDICT_AMBER_PP = 5.0, 15.0
_HELD_STATUSES = (20, 30, 40, 50)  # RentableItemStatus: held by a tenant (notice/vacant-rented/occupied)
_VACANT_STATUS = 10               # Vacant Unrented — assignable now
_UNRENTABLE_STATUS = 60
_TTL = 3600
_CACHE: dict = {}
_LOCK = threading.Lock()

OFFERS: dict[str, dict] = {
    "free_months_1": {"label": "One month free", "structure": "one_time_credit",
                      "outreach_offer_type": "free_months_1"},
    "rent_discount_5pct": {"label": "5% off rent, 12-month term-limited",
                           "structure": "recurring_discount_term",
                           "outreach_offer_type": "rent_discount_pct"},
    "free_garage_new": {"label": "Free garage — new assignment from the vacant pool",
                        "structure": "waiver", "outreach_offer_type": "free_garage_new"},
    "free_garage_waiver": {"label": "Free garage — waive the stall(s) already rented",
                           "structure": "waiver", "outreach_offer_type": "free_garage_waiver"},
}


# ---------------------------------------------------------------- pure economics
def _breakeven_pp(p: float, C: float | None, tc: float | None) -> float | None:
    if C is None or tc is None or tc <= 0 or C <= 0 or C >= tc:
        return None
    return round((1.0 - p) * C / (tc - C) * 100.0, 1)


def _verdict(be: float | None, C: float | None, tc: float | None) -> str:
    if C is None or tc is None or tc <= 0:
        return "n/a"
    if C >= tc:
        return "never"
    if be is None:
        return "n/a"
    return "green" if be <= VERDICT_GREEN_PP else ("amber" if be <= VERDICT_AMBER_PP else "red")


def _c_max(p: float, d: float, tc: float | None) -> float | None:
    if tc is None or tc <= 0:
        return None
    return round(d * tc / (1.0 - p + d), 0)


def offer_economics(p: float, tc_net: float | None, rent: float | None, has_garage: bool,
                    garage_rent_monthly: float = 0.0, vacant_stalls: int = 0,
                    vacant_avg_rent: float | None = None,
                    renewal_increase_pct: float | None = None,
                    assumed_uplift: float | None = None,
                    months_to_rerent: float = GARAGE_MONTHS_TO_RERENT) -> dict:
    """The per-resident offer table. PURE (no I/O) — this is what the offline test pins.
    Costs are 'if renewed' over one 12-month term; breakevens need no uplift; when
    `assumed_uplift` is passed the SCENARIO fields are added and labeled as such."""
    p = float(p)
    tc = float(tc_net) if tc_net else None
    rent = float(rent or 0.0)
    growth = ((renewal_increase_pct if renewal_increase_pct is not None
               else DEFAULT_RENT_GROWTH_PCT) / 100.0)
    offers: dict[str, dict] = {}

    def _mk(key: str, C: float | None, applicable: bool, reason: str | None = None, **extra):
        ok = bool(applicable and C is not None and C > 0 and tc)
        be = _breakeven_pp(p, C, tc) if ok else None
        row = {"label": OFFERS[key]["label"], "structure": OFFERS[key]["structure"],
               "outreach_offer_type": OFFERS[key]["outreach_offer_type"],
               "applicable": ok,
               "cost_if_renewed": round(C, 0) if C is not None else None,
               "breakeven_uplift_pp": be,
               "verdict": _verdict(be, C, tc) if ok else "n/a", **extra}
        if not ok:
            row["not_applicable_reason"] = reason or ("no rent or turnover cost on file"
                                                      if not (C and tc) else None)
        if assumed_uplift is not None and ok:
            net = assumed_uplift * tc - (1.0 - p + assumed_uplift) * C
            row["expected_net_at_assumed_uplift"] = round(net, 0)
            row["pays_at_assumed_uplift"] = bool(net > 0)
        offers[key] = row

    _mk("free_months_1", rent if rent > 0 else None, rent > 0)
    c5 = DISCOUNT_PCT / 100.0 * 12.0 * rent
    base3 = sum(DISCOUNT_PCT / 100.0 * rent * 12.0 * (1.0 + growth) ** y
                for y in range(DISCOUNT_BASE_YEARS))
    _mk("rent_discount_5pct", c5 if rent > 0 else None, rent > 0,
        equivalent_free_months=round(DISCOUNT_PCT / 100.0 * 12.0, 2),
        cost_if_baked_into_base_3yr=round(base3, 0) if rent > 0 else None,
        rent_growth_pct_assumed=round(growth * 100.0, 2))
    if has_garage:
        _mk("free_garage_new", None, False,
            reason="resident already rents a garage — see free_garage_waiver")
    elif not vacant_stalls:
        _mk("free_garage_new", None, False,
            reason="no vacant stall at this property right now (capacity)")
    elif not vacant_avg_rent:
        _mk("free_garage_new", None, False,
            reason="no garage inventory / stall rent on file for this property")
    else:
        cg = max(12.0 - float(months_to_rerent), 0.0) * float(vacant_avg_rent)
        _mk("free_garage_new", cg, True, tier_rent=round(float(vacant_avg_rent), 2),
            months_to_rerent_assumed=float(months_to_rerent),
            vacant_stalls_now=int(vacant_stalls))
    if has_garage and garage_rent_monthly and garage_rent_monthly > 0:
        _mk("free_garage_waiver", 12.0 * float(garage_rent_monthly), True,
            garage_rent_monthly=round(float(garage_rent_monthly), 2))
    else:
        _mk("free_garage_waiver", None, False,
            reason="resident does not rent a garage — see free_garage_new")

    c_max = {"at_5pp": _c_max(p, 0.05, tc), "at_10pp": _c_max(p, 0.10, tc),
             "at_15pp": _c_max(p, 0.15, tc)}
    live = {k: v for k, v in offers.items()
            if v["applicable"] and v["breakeven_uplift_pp"] is not None}
    cheapest = min(live, key=lambda k: live[k]["breakeven_uplift_pp"]) if live else None
    rec = None
    if assumed_uplift is not None:
        pos = {k: v["expected_net_at_assumed_uplift"] for k, v in offers.items()
               if v.get("pays_at_assumed_uplift")}
        rec = max(pos, key=pos.get) if pos else "none"
    return {"offers": offers, "c_max": c_max, "cheapest_breakeven_offer": cheapest,
            "recommended_offer_at_assumed_uplift": rec}


# ---------------------------------------------------------------- data lookups
def _garage(pids: list[int]) -> dict:
    """Garage/parking rentable items per property: stall inventory, what is vacant now
    (assignable), the tier rents, and which residents (TENANT.HMYPERSON) hold stalls.
    Count STALLS, never payers (Phase 0: 20 of 92 holders at 62W rent 2+ stalls)."""
    key = ("garage", tuple(pids))
    hit = _CACHE.get(key)
    if hit and time.time() - hit[0] < _TTL:
        return hit[1]
    ids = ",".join(str(int(p)) for p in pids)
    by_person: dict[int, dict] = {}
    by_prop: dict[int, dict] = {int(p): {"stalls": 0, "vacant_now": 0, "held": 0,
                                         "tier_rents": set(), "_vac": [], "_occ": []}
                                for p in pids}
    try:
        df = db.query(f"""
            SELECT ri.HPROP pid, ri.hTenant hmy, ri.iStatus st,
                   CAST(ri.CRENT AS FLOAT) rent, LTRIM(RTRIM(rt.SCODE)) typ
            FROM [{db.DB_YARDI}].dbo.RENTABLEITEMS ri
            JOIN [{db.DB_YARDI}].dbo.RENTABLEITEMSTYPE rt ON rt.HMY = ri.HRENTABLEITEMTYPE
            WHERE ri.HPROP IN ({ids}) AND rt.HCHARGECODE = {int(GARAGE_CHARGE_CODE)}""")
    except Exception:  # noqa: BLE001 — garage context is additive, never fails the view
        df = pd.DataFrame()
    for r in df.to_dict("records"):
        pid = int(r["pid"])
        st = int(r["st"] or 0)
        rent = float(r["rent"] or 0.0)
        bp = by_prop.setdefault(pid, {"stalls": 0, "vacant_now": 0, "held": 0,
                                      "tier_rents": set(), "_vac": [], "_occ": []})
        if st == _UNRENTABLE_STATUS:
            continue
        bp["stalls"] += 1
        if rent > 0:
            bp["tier_rents"].add(round(rent, 2))
        if st == _VACANT_STATUS:
            bp["vacant_now"] += 1
            if rent > 0:
                bp["_vac"].append(rent)
        elif st in _HELD_STATUSES:
            bp["held"] += 1
            if rent > 0:
                bp["_occ"].append(rent)
            hmy = r.get("hmy")
            if hmy is not None and not pd.isna(hmy) and int(hmy) != 0:
                e = by_person.setdefault(int(hmy), {"stalls_n": 0, "garage_rent_monthly": 0.0})
                e["stalls_n"] += 1
                e["garage_rent_monthly"] = round(e["garage_rent_monthly"] + rent, 2)
    for pid, bp in by_prop.items():
        vac, occ = bp.pop("_vac"), bp.pop("_occ")
        bp["vacant_avg_rent"] = (round(float(np.mean(vac)), 2) if vac
                                 else (round(float(np.mean(occ)), 2) if occ else None))
        bp["occupied_avg_rent"] = round(float(np.mean(occ)), 2) if occ else None
        bp["occupancy_pct"] = (round(100.0 * bp["held"] / bp["stalls"], 1)
                               if bp["stalls"] else None)
        bp["tier_rents"] = sorted(bp["tier_rents"])
    out = {"by_person": by_person, "by_property": by_prop,
           "source": "RENTABLEITEMS x RENTABLEITEMSTYPE (HCHARGECODE 17 = garage/parking); "
                     "status 10 = vacant-unrented (assignable now), 20/30/40/50 = held by a "
                     "tenant (hTenant -> TENANT.HMYPERSON); 60 unrentable excluded"}
    _CACHE[key] = (time.time(), out)
    return out


def _units(hmys: list[int]) -> dict[int, dict]:
    """Bedroom / sqft / market rent for each resident's CURRENT unit (TENANT.HUNIT -> UNIT)."""
    out: dict[int, dict] = {}
    ids_all = sorted({int(h) for h in hmys if h is not None and not pd.isna(h)})
    for i in range(0, len(ids_all), 500):
        ids = ",".join(str(h) for h in ids_all[i:i + 500])
        try:
            df = db.query(f"""
                SELECT t.HMYPERSON hmy, t.ISTATUS ist, u.IBEDROOMS bed,
                       CAST(u.DSQFT AS FLOAT) sqft, CAST(u.SRENT AS FLOAT) market_rent,
                       LTRIM(RTRIM(u.SCODE)) unit_code, u.EXCLUDE excl
                FROM [{db.DB_YARDI}].dbo.TENANT t
                JOIN [{db.DB_YARDI}].dbo.UNIT u ON u.HMY = t.HUNIT
                WHERE t.HMYPERSON IN ({ids})""")
        except Exception:  # noqa: BLE001
            continue
        for r in df.to_dict("records"):
            h = int(r["hmy"])
            cur = out.get(h)
            # prefer the active tenancy row if a person has several
            if cur is not None and cur.get("_ist") == 0:
                continue
            mr = r.get("market_rent")
            out[h] = {"bedrooms": int(r["bed"]) if pd.notna(r.get("bed")) else None,
                      "sqft": (round(float(r["sqft"]), 0)
                               if pd.notna(r.get("sqft")) and float(r["sqft"] or 0) > 0 else None),
                      "market_rent": (round(float(mr), 0)
                                      if mr is not None and not pd.isna(mr) and float(mr) > 200
                                      else None),
                      "unit_code": r.get("unit_code"),
                      "unit_excluded": bool(int(r.get("excl") or 0) != 0),
                      "_ist": int(r.get("ist") or 0) if pd.notna(r.get("ist")) else None}
    for v in out.values():
        v.pop("_ist", None)
    return out


def _ledger(pids: list[int], asof: str | None = None) -> tuple[dict, dict, dict | None]:
    """Phase 1 turnover ledger indexed (property, bedroom) + property-level context,
    pinned to the cohort's as_of so a snapshot is reproducible."""
    try:
        led = unit_economics.build(pids, as_of=asof)
    except Exception:  # noqa: BLE001
        return {}, {}, None
    by_bed: dict[tuple[int, int], dict] = {}
    by_prop: dict[int, dict] = {}
    for pr in led.get("properties", []):
        pid = int(pr["property_id"])
        nets = []
        for b in pr.get("bedrooms", []):
            by_bed[(pid, int(b["bedrooms"]))] = b
            if b.get("turnover_cost_net"):
                nets.append(float(b["turnover_cost_net"]))
        ms = pr.get("make_ready_source") or {}
        by_prop[pid] = {"renewal_increase_pct": pr.get("renewal_increase_pct"),
                        "make_ready_measured": bool(ms.get("measured")),
                        "gl_coverage": ms.get("gl_coverage"),
                        "annual_turn_cost_total": pr.get("annual_turn_cost_total"),
                        "tc_net_property_mean": round(float(np.mean(nets)), 0) if nets else None}
    return by_bed, by_prop, led


def _duration(g: pd.DataFrame) -> dict[str, list]:
    """INTERNAL duration-model estimates (v0.1, report-only, unvalidated in production):
    `full` = expected remaining months given today's lease state; `trait` = stayer-type
    score with lease-timing features removed. Neither is E[terms | saved]; they are the
    inputs the intern needs to reason about a multi-term horizon (handoff §5.3)."""
    out = {"full": [None] * len(g), "trait": [None] * len(g)}
    try:
        from contracts.multifamily_churn import edge
        b = edge._duration_bundle()
        cap = float(b.get("cap_months") or 48.0)
        for name in ("full", "trait"):
            v = b["variants"][name]
            X = pd.DataFrame({c: pd.to_numeric(g.get(c), errors="coerce") for c in v["features"]})
            pred = np.clip(v["model"].predict(X), 0, cap)
            out[name] = [round(float(x), 1) for x in pred]
    except Exception:  # noqa: BLE001
        pass
    return out


def _model_input(g: pd.DataFrame, b: dict) -> pd.DataFrame | None:
    """EXACTLY what the bundle scores: prepare_v4 output (imputed from the bundle's
    training medians, property/pet label-encoded, bundle feature order)."""
    try:
        return prepare_v4.prepare(g, b)
    except Exception:  # noqa: BLE001
        return None


def _drivers(g: pd.DataFrame, b: dict, X: pd.DataFrame | None) -> list[dict]:
    """Per-resident why/drivers from the h180 bundle's OWN booster (persuadables uses the
    h60 serving path for these, which explains a different score)."""
    empty = [{"why": None, "drivers": []} for _ in range(len(g))]
    if X is None:
        return empty
    try:
        Xs = b["scaler"].transform(X)
        contrib = explain.contributions(b, Xs)
        if contrib is None:
            return empty
        props = g.get("PropertyId")
        out = []
        for i in range(len(g)):
            pid = props.iloc[i] if props is not None else None
            pname = data.PROPERTIES.get(int(pid)) if pid is not None and pd.notna(pid) else None
            out.append(explain.why(b, X, contrib, i, top_k=5, prop_name=pname))
        return out
    except Exception:  # noqa: BLE001
        return empty


def _num(v, nd: int = 2):
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v):
            return None
        return round(float(v), nd)
    except (TypeError, ValueError):
        return None


def _d(v):
    s = str(v or "")[:10]
    return s or None


# ---------------------------------------------------------------- the view
def _empty(meta: dict, pop: dict, summary: str) -> dict:
    """Same shape as a populated response so callers never branch on missing keys."""
    return {**meta, "population": pop, "n": 0, "shown": 0,
            "cohort": {"n": 0, "expected_moveouts": 0.0, "expected_loss": 0,
                       "monthly_rent_at_risk": 0, "avg_churn": None, "by_property": [],
                       "by_bedroom": [], "cheapest_breakeven_offer_counts": {}},
            "offers_summary": {}, "context": {}, "members": [], "summary": summary}


def build(property_ids: list[int] | None = None, as_of: str | None = None,
          band_low: float = BAND_LOW, band_high: float = BAND_HIGH,
          decision_window_days: int | None = DECISION_WINDOW_DAYS,
          include_notice: bool = False, assumed_uplift: float | None = None,
          garage_months_to_rerent: float = GARAGE_MONTHS_TO_RERENT,
          top: int | None = None) -> dict:
    if not (0.0 <= band_low < band_high <= 1.0):
        raise ValueError(f"band must satisfy 0 <= low < high <= 1 (got {band_low}, {band_high})")
    if assumed_uplift is not None and not (0.0 < assumed_uplift < 1.0):
        raise ValueError(f"assumed_uplift must be in (0, 1) as a probability reduction "
                         f"(got {assumed_uplift})")
    if decision_window_days is not None and int(decision_window_days) < 0:
        raise ValueError("decision_window_days must be >= 0 (or omitted for no window)")
    if not (0.0 <= float(garage_months_to_rerent) <= 12.0):
        raise ValueError("garage_months_to_rerent must be within [0, 12]")
    pids = sorted(int(p) for p in (property_ids or data.DEFAULT_PROPERTY_IDS))
    df, b = cohorts._scored_frame(pids, as_of, version=RENEWAL_BUNDLE)
    asof = data._safe_asof(as_of)
    window = int(decision_window_days) if decision_window_days is not None else None

    meta = {
        "contract": "multifamily-churn", "view": "concession-cohort",
        "model_version": b.get("version"), "horizon_days": b.get("horizon_days"),
        "as_of": asof,
        "properties": [{"property_id": p, "name": data.PROPERTIES.get(p)} for p in pids],
        "definition": {
            "band": {"low": band_low, "high": band_high,
                     "note": "calibrated churn probability at the NEXT lease decision, "
                             "h180 renewal-forecast bundle, unclipped; stay = 1 - churn"},
            "decision_window_days": window,
            "notice_filed": "included" if include_notice else "excluded",
            "pii": "identifier-only rows (scode / hmyperson / unit / property); no names"},
        "assumptions": {
            "uplift": ("SCENARIO — assumed_uplift=%.3f supplied by caller; expected_net_* "
                       "fields are scenario math, not measurement" % assumed_uplift)
                      if assumed_uplift is not None else
                      "NONE assumed — breakeven_uplift_pp is the assumption-free read "
                      "(history supports single digits inside a ±5–13pp confounding floor)",
            "garage_new_cost": f"(12 − {garage_months_to_rerent:g} months to re-rent) × the "
                               "vacant stall's rent; applicable only without a garage and "
                               "with a vacant stall at the property",
            "garage_waiver_cost": "12 × the stall rent the resident pays now",
            "discount_cost": f"{DISCOUNT_PCT:g}% × 12 × rent term-limited (= "
                             f"{DISCOUNT_PCT / 100 * 12:.1f} free months); "
                             "cost_if_baked_into_base_3yr compounds at the property's "
                             "measured renewal increase",
            "turnover_cost": "TC_net per bedroom × property from /multifamily/unit-economics "
                             "(vacancy × in-place rent + make-ready − net relet premium); "
                             "property-mean fallback when the bedroom is unknown",
            "duration_internal": "duration bundle v0.1 (INTERNAL, report-only, unvalidated "
                                 "in production): full = expected remaining months given "
                                 "today's lease state; trait = stayer-type score without "
                                 "lease-timing features. Neither is E[terms | saved]",
            "features": "the exact model input: bundle feature order, imputed from training "
                        "medians, PropertyName/PrimaryPetType label-encoded",
            "verdict_buckets": {"green": f"<= {VERDICT_GREEN_PP:g}pp",
                                "amber": f"<= {VERDICT_AMBER_PP:g}pp",
                                "red": f"> {VERDICT_AMBER_PP:g}pp",
                                "never": "cost >= turnover cost"}},
        "formulas": {
            "expected_loss": "p × TC_net",
            "expected_net_gain": "Δ × TC_net − (1 − p + Δ) × C",
            "breakeven_uplift_pp": "(1 − p) × C / (TC_net − C) × 100",
            "c_max": "Δ × TC_net / (1 − p + Δ)"},
    }
    if df.empty:
        return _empty(meta, {"scored_n": 0}, "No active residents found.")

    notice = pd.to_numeric(df.get("HasNotice"), errors="coerce").fillna(0) == 1
    in_band = df["p_churn"].between(band_low, band_high, inclusive="left")
    d2e = pd.to_numeric(df.get("_d2e"), errors="coerce")
    in_window = (d2e <= window) if window is not None else pd.Series(True, index=df.index)
    in_window = in_window.fillna(False)
    not_notice = (~notice) if not include_notice else pd.Series(True, index=df.index)
    keep = in_band & in_window & not_notice
    g = df[keep].copy().reset_index(drop=True)

    pop = {"scored_n": int(len(df)), "avg_churn": round(float(df["p_churn"].mean()), 3),
           "in_band_n": int(in_band.sum()),
           "in_band_excluded_notice_n": int((in_band & notice).sum()) if not include_notice else 0,
           "in_band_excluded_window_n": int((in_band & ~in_window & not_notice).sum()),
           "below_band_n": int((df["p_churn"] < band_low).sum()),
           "above_band_n": int((df["p_churn"] >= band_high).sum()),
           "notice_given_n": int(notice.sum())}
    if g.empty:
        return _empty(meta, pop,
                      f"No residents sit in the {band_low:g}-{band_high:g} churn band"
                      f"{'' if window is None else f' with a decision within {window} days'}"
                      f"{'' if include_notice else ' without notice'} out of {len(df)} scored "
                      f"(no-notice p99 today: {float(df.loc[~notice, 'p_churn'].quantile(0.99)):.2f}"
                      f" — widen band_low).")

    # ---- enrichment (each block is additive and never fails the view) ----
    units = _units(g["hMyPerson"].tolist()) if "hMyPerson" in g.columns else {}
    by_bed, by_prop, led = _ledger(pids, asof)
    gar = _garage(pids)
    dur = _duration(g)
    X_in = _model_input(g, b)
    expl = _drivers(g, b, X_in)
    classes = persuadables._cohort_classes(df)
    try:
        from contracts.multifamily_churn import incentives
        inc = incentives.for_scodes([str(s).strip() for s in g["SCODE"]])
    except Exception:  # noqa: BLE001
        inc = {}
    lo, mid = contract._tiers(b)
    feat_cols = list(X_in.columns) if X_in is not None else [
        c for c in (b.get("feature_columns") or data.MODEL_FEATURES) if c in g.columns]

    members = []
    for i, r in enumerate(g.to_dict("records")):
        scode = str(r.get("SCODE", "")).strip()
        hmy = int(r["hMyPerson"]) if pd.notna(r.get("hMyPerson")) else None
        pid = int(r["PropertyId"]) if pd.notna(r.get("PropertyId")) else None
        p = float(r["p_churn"])
        u = units.get(hmy, {}) if hmy is not None else {}
        bed = u.get("bedrooms")
        rent = _num(r.get("CurrentRent"))
        if not rent or rent <= 0:
            rent = _num(r.get("ActiveRent"))
        rent = rent if rent and rent > 0 else None
        mkt = u.get("market_rent")
        led_row = by_bed.get((pid, bed)) if (pid is not None and bed is not None) else None
        pctx = by_prop.get(pid, {}) if pid is not None else {}
        if led_row is not None:
            tc_net, tc_gross = led_row.get("turnover_cost_net"), led_row.get("turnover_cost_gross")
            rate, mk_method = led_row.get("turnover_rate_per_unit_yr"), led_row.get("make_ready_method")
            tc_source = "ledger: property x bedroom"
        else:
            tc_net, tc_gross, rate = pctx.get("tc_net_property_mean"), None, None
            mk_method = "property-mean fallback (bedroom unknown or not in ledger)"
            tc_source = "ledger: property mean across bedrooms" if tc_net else "unavailable"
        gp = gar["by_person"].get(hmy, {}) if hmy is not None else {}
        gprop = gar["by_property"].get(pid, {}) if pid is not None else {}
        has_garage = bool(gp.get("stalls_n", 0) > 0)
        econ = offer_economics(p, tc_net, rent, has_garage,
                               garage_rent_monthly=gp.get("garage_rent_monthly", 0.0),
                               vacant_stalls=gprop.get("vacant_now", 0),
                               vacant_avg_rent=gprop.get("vacant_avg_rent"),
                               renewal_increase_pct=pctx.get("renewal_increase_pct"),
                               assumed_uplift=assumed_uplift,
                               months_to_rerent=garage_months_to_rerent)
        d2 = int(r["_d2e"]) if pd.notna(r.get("_d2e")) else None
        members.append({
            "scode": scode, "hmyperson": hmy,
            "unit": (str(r.get("UnitCode")).strip() if r.get("UnitCode") is not None
                     else u.get("unit_code")),
            "property_id": pid, "property": data.PROPERTIES.get(pid) if pid else None,
            "bedrooms": bed, "sqft": u.get("sqft"),
            "lease": {"lease_from": _d(r.get("LeaseFromDate")), "lease_to": _d(r.get("LeaseToDate")),
                      "days_to_lease_end": d2, "holdover": (d2 is not None and d2 < 0),
                      "lease_term_months": _num(r.get("LeaseTermMonths"), 0),
                      "decision_number": _num(r.get("DecisionNumber"), 0),
                      "tenure_months": _num(r.get("TenureMonths"), 1),
                      "move_in": _d(r.get("MoveInDate")),
                      "has_notice": int(pd.to_numeric(pd.Series([r.get("HasNotice")]),
                                                      errors="coerce").fillna(0).iloc[0])},
            "rent": {"monthly_rent": rent, "prior_rent": _num(r.get("PriorRent")),
                     "rent_increase_pct": _num(r.get("RentIncreasePct")),
                     "market_rent": mkt,
                     "rent_vs_market_pct": (round((rent - mkt) / mkt * 100.0, 2)
                                            if rent and mkt else None)},
            "model": {"churn_probability": round(p, 4), "stay_probability": round(1.0 - p, 4),
                      "risk_tier": contract._tier(p, lo, mid),
                      "why": expl[i].get("why"), "drivers": expl[i].get("drivers") or []},
            "cohorts": persuadables._memberships(pd.Series(r), classes),
            "garage": {"has_garage": has_garage, "stalls_n": int(gp.get("stalls_n", 0)),
                       "garage_rent_monthly": _num(gp.get("garage_rent_monthly", 0.0))},
            "duration_internal": {"expected_remaining_months_full": dur["full"][i],
                                  "stayer_trait_months": dur["trait"][i]},
            "incentives": inc.get(scode.lower()),
            "economics": {"turnover_cost_net": tc_net, "turnover_cost_gross": tc_gross,
                          "turnover_cost_source": tc_source,
                          "turnover_rate_per_unit_yr": rate, "make_ready_method": mk_method,
                          "expected_loss": round(p * tc_net, 0) if tc_net else None,
                          "monthly_rent_at_risk": round(p * rent, 0) if rent else None,
                          **econ},
            "features": ({c: _num(X_in.iloc[i][c], 4) for c in feat_cols} if X_in is not None
                         else {c: _num(r.get(c), 4) for c in feat_cols}),
        })
    members.sort(key=lambda m: (-(m["economics"]["expected_loss"] or 0), m["scode"]))
    rows_all = members
    full_n = len(rows_all)
    shown = rows_all if top is None else rows_all[:max(int(top), 0)]

    # ---- cohort roll-ups (over the WHOLE cohort, never the displayed slice) ----
    exp_moveouts = float(g["p_churn"].sum())

    def _agg(rows: list[dict]) -> dict:
        return {"n": len(rows),
                "expected_moveouts": round(sum(m["model"]["churn_probability"] for m in rows), 1),
                "expected_loss": round(sum(m["economics"]["expected_loss"] or 0 for m in rows), 0),
                "monthly_rent_at_risk": round(sum(m["economics"]["monthly_rent_at_risk"] or 0
                                                  for m in rows), 0)}
    by_property = {}
    by_bedroom = {}
    for m in rows_all:
        by_property.setdefault(m["property_id"], []).append(m)
        by_bedroom.setdefault(m["bedrooms"], []).append(m)
    offers_summary: dict[str, dict] = {}
    for key in OFFERS:
        rows_k = [m["economics"]["offers"][key] for m in rows_all]
        app = [o for o in rows_k if o["applicable"]]
        bes = [o["breakeven_uplift_pp"] for o in app if o["breakeven_uplift_pp"] is not None]
        verdicts = {}
        for o in rows_k:
            verdicts[o["verdict"]] = verdicts.get(o["verdict"], 0) + 1
        s = {"label": OFFERS[key]["label"], "applicable_n": len(app),
             "total_cost_if_all_renew": round(sum(o["cost_if_renewed"] or 0 for o in app), 0),
             "median_breakeven_pp": round(float(np.median(bes)), 1) if bes else None,
             "min_breakeven_pp": round(float(np.min(bes)), 1) if bes else None,
             "verdicts": verdicts}
        if assumed_uplift is not None:
            nets = [o.get("expected_net_at_assumed_uplift") for o in app
                    if o.get("expected_net_at_assumed_uplift") is not None]
            s["expected_net_total_at_assumed_uplift"] = round(sum(nets), 0) if nets else None
            s["pays_n_at_assumed_uplift"] = sum(1 for o in app if o.get("pays_at_assumed_uplift"))
        offers_summary[key] = s
    cheapest_counts: dict[str, int] = {}
    for m in rows_all:
        k = m["economics"]["cheapest_breakeven_offer"] or "none"
        cheapest_counts[k] = cheapest_counts.get(k, 0) + 1

    cohort = {**_agg(rows_all),
              "avg_churn": round(float(g["p_churn"].mean()), 3),
              "by_property": [{"property_id": pid, "name": data.PROPERTIES.get(pid),
                               **_agg(rows)} for pid, rows in sorted(by_property.items(),
                                                                    key=lambda kv: (kv[0] is None, kv[0]))],
              "by_bedroom": [{"bedrooms": bed, **_agg(rows)}
                             for bed, rows in sorted(by_bedroom.items(),
                                                     key=lambda kv: (kv[0] is None, kv[0]))],
              "cheapest_breakeven_offer_counts": cheapest_counts}
    context = {
        "ledger": {pid: v for pid, v in by_prop.items()},
        "garage_capacity": {pid: v for pid, v in gar["by_property"].items()},
        "garage_source": gar["source"],
        "portfolio_annual_turn_cost": led.get("portfolio_annual_turn_cost") if led else None,
        "ledger_window_years": led.get("window_years") if led else None}

    scope = ", ".join(data.PROPERTIES.get(p, str(p)) for p in pids)
    win_txt = "" if window is None else f", decision within {window} days"
    best = max(cheapest_counts.items(), key=lambda kv: kv[1])[0] if cheapest_counts else "none"
    summary = (
        f"{full_n} residents sit in the {band_low:g}-{band_high:g} churn band at the "
        f"{b.get('horizon_days')}-day renewal decision ({'notice included' if include_notice else 'no notice'}"
        f"{win_txt}) across {scope}, out of {len(df)} scored: {exp_moveouts:.1f} expected "
        f"move-outs worth ${cohort['expected_loss']:,.0f} in turns if we do nothing. "
        f"Assumption-free read: the lowest-breakeven offer is '{best}' for "
        f"{cheapest_counts.get(best, 0)} of them; no uplift is assumed"
        + (f" (scenario at {assumed_uplift:.0%} uplift: "
           f"{sum(1 for m in rows_all if m['economics']['recommended_offer_at_assumed_uplift'] not in (None, 'none'))} "
           f"residents have an offer that pays)." if assumed_uplift is not None else "."))

    return {**meta, "population": pop, "n": full_n, "shown": len(shown),
            "cohort": cohort, "offers_summary": offers_summary, "context": context,
            "members": shown, "summary": summary}


# ---------------------------------------------------------------- CSV export
def _fmt_item(x) -> str:
    if isinstance(x, dict):
        if "feature" in x:
            sign = "+" if x.get("direction") == "up" else "-"
            return f"{x['feature']}{sign}{x.get('weight')}"
        if "cohort" in x:
            return str(x["cohort"])
        return json.dumps(x, sort_keys=True, default=str)
    return str(x)


def _flatten(d: dict, prefix: str = "", out: dict | None = None) -> dict:
    out = {} if out is None else out
    for k, v in d.items():
        kk = f"{prefix}{k}"
        if isinstance(v, dict):
            _flatten(v, kk + ".", out)
        elif isinstance(v, (list, tuple)):
            out[kk] = "|".join(_fmt_item(x) for x in v)
        else:
            out[kk] = v
    return out


def to_csv(payload: dict) -> str:
    """One row per resident, nested blocks flattened to dotted columns (economics.offers.
    free_months_1.breakeven_uplift_pp, features.TenureMonths, ...). Lists join with '|'."""
    rows = [_flatten(m) for m in payload.get("members", [])]
    cols: list[str] = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                cols.append(k)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({k: ("" if v is None else v) for k, v in r.items()})
    return buf.getvalue()
