"""Per-unit-type (bedroom) turnover economics — the decision ledger (Phase 1).

Prices what one turn actually costs, per bedroom, per property, from live Yardi:

  turnover_cost_gross = vacancy_months x avg in-place rent + make-ready
  turnover_cost_net   = gross - re-lease premium net of the renewal increase (12 mo)
  turnover_rate       = turns / occupied unit-years   (per-unit INTERVAL UNION —
                        summing tenancy spans double-counts overlapping co-tenancies;
                        Phase 0 review 2026-08-18 found ~104 phantom unit-years at 62W)
  expected_loss/unit-yr = turnover_rate x turnover_cost_net

Make-ready is MEASURED where the property's GL carries the `U=<unit>` note convention
(GLDETAIL.SNOTES, IBOOK=1 accrual only) at sufficient coverage: per-turn direct cost
(median of paint/janitorial/carpet + turnover-replacement accounts, invoices attributed
to the nearest turn within [move-out - 7d, next move-in + 45d]) plus a property-level
pool (make-ready stock/supplies/locks/vacant electricity) allocated flat per turn.
The `U=` convention is a local bookkeeping habit (validated at 62W, 2026-08-18) — where
coverage is thin the module falls back to a flat $600 labeled as such. Measured figures
are contracted-vendor + materials ONLY: in-house maintenance payroll has no make-ready
split and is excluded (stated in `methods`).

Renewal increases come from Lease_History successive executed terms (iStatus 4/5,
cRent deltas) — the point-in-time rent TENANT cannot give. Everything carries n's and
a method label; nothing here assumes an uplift, a flip rate, or a model score.

Full derivation + landmines: docs/PHASE0_DISCOVERY_62W_2026-08-18.md.
"""
from __future__ import annotations

import re
import threading
import time

import pandas as pd

from core import db
from contracts.multifamily_churn import data

_CACHE: dict = {}
_LOCK = threading.Lock()
_TTL = 24 * 3600

WINDOW_YEARS = 4
SHRINK_K = 20.0                 # empirical-Bayes weight toward the property mean
_AVG_DAYS_PER_MONTH = 30.44
FALLBACK_MAKE_READY = 600.0     # legacy flat placeholder, used only when GL coverage fails
_COVERAGE_MIN = 0.50            # share of turns with attributed GL cost to trust "measured"
_RELET_NET_TERM_MONTHS = 12

# GL account census (2026-08-18, property 82; the chart is shared portfolio-wide).
# Direct = unit-attributable via the U= note; excludes life-cycle capex (87xxx/88xxx
# beyond the three explicitly named "Turnover") to avoid double-counting capital into
# a retention decision. Pool = real turn spend with no unit tag -> flat per-turn adder.
DIRECT_ACCTS = ("52501002", "52501004", "52201002", "52251003",
                "87000003", "87000005", "87000006")
POOL_ACCTS = ("52141005", "52520000", "52330000", "52340000", "55052000")
_UNIT_NOTE = re.compile(r"^U=\s*([0-9A-Za-z]+)")


def _turns_frame(pid: int, asof: str | None = None) -> pd.DataFrame:
    """Turn events (move-out -> next move-in on the same unit, gap 0-365d, move-out in
    the trailing window ending at `asof`), with bedroom, unit code, and same-unit relet
    rent delta. Only occupancy statuses (data.OCC_SQL): canceled/denied application rows
    otherwise pose as the "next move-in" and truncate the vacancy (2026-09-25 audit)."""
    asof = data._safe_asof(asof)
    df = db.query(f"""
        SELECT u.IBEDROOMS bed, LTRIM(RTRIM(u.SCODE)) ucode, g.DTMOVEOUT, g.next_in,
               g.gap, g.delta_pct
        FROM (
          SELECT t.HUNIT, t.DTMOVEOUT,
                 LEAD(t.DTMOVEIN) OVER (PARTITION BY t.HUNIT ORDER BY t.DTMOVEIN) next_in,
                 DATEDIFF(DAY, t.DTMOVEOUT,
                          LEAD(t.DTMOVEIN) OVER (PARTITION BY t.HUNIT ORDER BY t.DTMOVEIN)) gap,
                 CASE WHEN t.SRENT > 200 THEN
                   (LEAD(t.SRENT) OVER (PARTITION BY t.HUNIT ORDER BY t.DTMOVEIN) - t.SRENT)
                     / NULLIF(t.SRENT, 0) * 100.0 END delta_pct
          FROM [{db.DB_YARDI}].dbo.TENANT t
          WHERE t.HPROPERTY = ? AND t.DTMOVEIN IS NOT NULL AND t.{data.OCC_SQL}
        ) g
        JOIN [{db.DB_YARDI}].dbo.UNIT u ON u.HMY = g.HUNIT
        WHERE g.gap BETWEEN 0 AND 365
          AND g.DTMOVEOUT >= DATEADD(YEAR, -{WINDOW_YEARS}, '{asof}') AND g.DTMOVEOUT <= '{asof}'
          AND u.EXCLUDE = 0 AND u.IBEDROOMS IS NOT NULL""", (pid,))
    if not df.empty:
        df["DTMOVEOUT"] = pd.to_datetime(df["DTMOVEOUT"])
        df["next_in"] = pd.to_datetime(df["next_in"])
        df["ucode"] = df["ucode"].astype(str).str.strip().str.lower()
    return df


def _occupied_unit_years(pid: int, asof: str | None = None) -> dict:
    """Occupied unit-years per bedroom over the window, as the per-unit UNION of
    tenancy intervals (co-tenant and overlapping bookkeeping rows must not double
    count). Returns {bed: unit_years}."""
    df = db.query(f"""
        SELECT u.IBEDROOMS bed, t.HUNIT, t.DTMOVEIN, t.DTMOVEOUT
        FROM [{db.DB_YARDI}].dbo.TENANT t
        JOIN [{db.DB_YARDI}].dbo.UNIT u ON u.HMY = t.HUNIT
        WHERE t.HPROPERTY = ? AND t.DTMOVEIN IS NOT NULL AND t.{data.OCC_SQL}
          AND u.EXCLUDE = 0 AND u.IBEDROOMS IS NOT NULL""", (pid,))
    if df.empty:
        return {}
    end = pd.Timestamp(data._safe_asof(asof))
    start = end - pd.DateOffset(years=WINDOW_YEARS)
    a = pd.to_datetime(df["DTMOVEIN"]).clip(lower=start)
    b = pd.to_datetime(df["DTMOVEOUT"]).fillna(end).clip(upper=end)
    df = df.assign(a=a, b=b)
    df = df[df["b"] > df["a"]]
    out: dict[int, float] = {}
    for (bed, _unit), g in df.groupby(["bed", "HUNIT"]):
        iv = sorted(zip(g["a"], g["b"]))
        cs, ce = iv[0]
        days = 0
        for s, e in iv[1:]:
            if s <= ce:
                ce = max(ce, e)
            else:
                days += (ce - cs).days
                cs, ce = s, e
        days += (ce - cs).days
        out[int(bed)] = out.get(int(bed), 0.0) + days / 365.25
    return out


def _mix_and_rents(pid: int) -> pd.DataFrame:
    """Rentable-unit mix + current avg in-place rent (active leases) and avg market
    rent per bedroom. In-place rent is the forward-looking vacancy-loss basis; TENANT
    current rows are current-state, which is exactly right here (never historical)."""
    return db.query(f"""
        SELECT u.IBEDROOMS bed, COUNT(DISTINCT u.HMY) units,
               AVG(CASE WHEN u.SRENT > 200 THEN CAST(u.SRENT AS FLOAT) END) market_rent,
               AVG(CASE WHEN t.ISTATUS = 0 AND t.SRENT > 200
                        THEN CAST(t.SRENT AS FLOAT) END) inplace_rent,
               SUM(CASE WHEN t.ISTATUS = 0 THEN 1 ELSE 0 END) active_residents,
               AVG(CASE WHEN u.DSQFT > 0 THEN CAST(u.DSQFT AS FLOAT) END) avg_sqft
        FROM [{db.DB_YARDI}].dbo.UNIT u
        LEFT JOIN [{db.DB_YARDI}].dbo.TENANT t ON t.HUNIT = u.HMY
        WHERE u.HPROPERTY = ? AND u.EXCLUDE = 0 AND u.IBEDROOMS IS NOT NULL
        GROUP BY u.IBEDROOMS""", (pid,))


def _renewal_increase_pct(pid: int, asof: str | None = None) -> float | None:
    asof = data._safe_asof(asof)
    """Typical renewal increase from Lease_History successive executed terms (point-in-
    time cRent, iStatus 4/5) — what we'd have charged the retained resident anyway;
    the relet premium only counts NET of this."""
    df = db.query(f"""
        SELECT AVG(pct) avg_pct, COUNT(*) n FROM (
          SELECT (LEAD(lh.cRent) OVER (PARTITION BY lh.hTent ORDER BY lh.dtLeaseFrom, lh.hMy)
                  - lh.cRent) / NULLIF(lh.cRent, 0) * 100.0 pct,
                 lh.dtLeaseTo
          FROM [{db.DB_YARDI}].dbo.Lease_History lh
          JOIN [{db.DB_YARDI}].dbo.TENANT t ON t.HMYPERSON = lh.hTent
          WHERE t.HPROPERTY = ? AND lh.iStatus IN (4, 5) AND lh.cRent > 200
        ) g
        WHERE pct IS NOT NULL AND pct BETWEEN -20 AND 30
          AND g.dtLeaseTo >= DATEADD(YEAR, -{WINDOW_YEARS}, '{asof}') AND g.dtLeaseTo <= '{asof}'""", (pid,))
    if df.empty or pd.isna(df.iloc[0]["avg_pct"]):
        return None
    return round(float(df.iloc[0]["avg_pct"]), 2)


def _gl_make_ready(pid: int, turns: pd.DataFrame, asof: str | None = None) -> dict:
    asof = data._safe_asof(asof)
    """Measured make-ready per bedroom: parse `U=<unit>` from GLDETAIL notes on the
    in-scope accounts (IBOOK=1 only — the cash book duplicates every expense), assign
    each invoice to the nearest turn within [move-out - 7d, next move-in + 45d], and
    take the per-bedroom MEDIAN (turn costs are heavy-tailed). Pool accounts have no
    unit tag and allocate flat per turn. Coverage below _COVERAGE_MIN -> caller falls
    back; a re-run per property is mandatory before trusting this off 62W."""
    accts = ",".join(f"'{a}'" for a in DIRECT_ACCTS)
    gl = db.query(f"""
        SELECT g.hMy gl_id, g.DTDATE, g.DAMOUNT, g.SNOTES
        FROM [{db.DB_YARDI}].dbo.GLDETAIL g
        JOIN [{db.DB_YARDI}].dbo.ACCT a ON a.HMY = g.HACCT
        WHERE g.HPROP = ? AND g.IBOOK = 1 AND g.SNOTES LIKE 'U=%'
          AND g.DTDATE >= DATEADD(YEAR, -{WINDOW_YEARS + 1}, '{asof}') AND g.DTDATE <= DATEADD(DAY, 45, '{asof}')
          AND LTRIM(RTRIM(a.SCODE)) IN ({accts})""", (pid,))
    pool = db.query(f"""
        SELECT SUM(g.DAMOUNT) amt
        FROM [{db.DB_YARDI}].dbo.GLDETAIL g
        JOIN [{db.DB_YARDI}].dbo.ACCT a ON a.HMY = g.HACCT
        WHERE g.HPROP = ? AND g.IBOOK = 1
          AND g.DTDATE >= DATEADD(YEAR, -{WINDOW_YEARS}, '{asof}') AND g.DTDATE <= '{asof}'
          AND LTRIM(RTRIM(a.SCODE)) IN ({",".join(f"'{a}'" for a in POOL_ACCTS)})""", (pid,))
    n_turns = len(turns)
    pool_amt = (float(pool.iloc[0]["amt"]) if not pool.empty
                and pd.notna(pool.iloc[0]["amt"]) else 0.0)
    pool_per_turn = round(pool_amt / n_turns, 0) if n_turns else 0.0
    out = {"coverage": 0.0, "pool_per_turn": pool_per_turn, "per_bed": {}, "n_costs": 0}
    if gl.empty or turns.empty:
        return out
    gl = gl.assign(ucode=gl["SNOTES"].astype(str).str.extract(_UNIT_NOTE, expand=False)
                   .str.strip().str.lower(),
                   DTDATE=pd.to_datetime(gl["DTDATE"]),
                   DAMOUNT=pd.to_numeric(gl["DAMOUNT"], errors="coerce").astype(float)
                   ).dropna(subset=["ucode", "DAMOUNT"])
    t = turns.reset_index(drop=True).reset_index(names="turn_id")
    m = gl.merge(t[["turn_id", "ucode", "bed", "DTMOVEOUT", "next_in"]], on="ucode")
    m = m[(m["DTDATE"] >= m["DTMOVEOUT"] - pd.Timedelta(days=7)) &
          (m["DTDATE"] <= m["next_in"] + pd.Timedelta(days=45))]
    if m.empty:
        return out
    # each GL line lands on its NEAREST turn only (no double counting)
    m["dist"] = (m["DTDATE"] - m["DTMOVEOUT"]).abs()
    m = m.sort_values("dist").groupby("gl_id", as_index=False).first()
    per_turn = m.groupby("turn_id")["DAMOUNT"].sum()
    cost = t.set_index("turn_id").join(per_turn.rename("cost")).fillna({"cost": 0.0})
    out["coverage"] = round(float((cost["cost"] > 0).mean()), 3)
    out["n_costs"] = int(len(m))
    for bed, g in cost.groupby("bed"):
        out["per_bed"][int(bed)] = {
            "median_direct": round(float(g["cost"].median()), 0),
            "mean_direct": round(float(g["cost"].mean()), 0),
            "n_turns": int(len(g)),
            "zero_cost_share": round(float((g["cost"] <= 0).mean()), 3)}
    return out


def _property_block(pid: int, asof: str) -> dict:
    turns = _turns_frame(pid, asof)
    mix = _mix_and_rents(pid)            # current state: in-place rent is not as-of-able
    occ = _occupied_unit_years(pid, asof)
    ri_pct = _renewal_increase_pct(pid, asof)
    gl = _gl_make_ready(pid, turns, asof)
    measured = gl["coverage"] >= _COVERAGE_MIN

    prop_days = None
    if not turns.empty:
        prop_days = float(turns["gap"].mean())

    rows = []
    for r in mix.to_dict("records"):
        bed = int(r["bed"])
        tb = turns[turns["bed"] == bed] if not turns.empty else pd.DataFrame()
        n = int(len(tb))
        raw_days = float(tb["gap"].mean()) if n else None
        shrunk = (None if raw_days is None or prop_days is None
                  else (n * raw_days + SHRINK_K * prop_days) / (n + SHRINK_K))
        vac_mo = round(shrunk / _AVG_DAYS_PER_MONTH, 2) if shrunk is not None else None
        rent = (float(r["inplace_rent"]) if pd.notna(r.get("inplace_rent"))
                else (float(r["market_rent"]) if pd.notna(r.get("market_rent")) else None))
        relet = (float(pd.to_numeric(tb["delta_pct"], errors="coerce").mean())
                 if n and tb["delta_pct"].notna().any() else None)
        if measured and bed in gl["per_bed"] and gl["per_bed"][bed]["n_turns"] >= 20:
            mk = gl["per_bed"][bed]["median_direct"] + gl["pool_per_turn"]
            mk_method = "measured: GL U= median direct + property pool/turn"
        elif measured and gl["per_bed"]:
            # thin bedroom (e.g. 3BR n=7): scale the largest measured bedroom by sqft
            ref_bed, ref = max(gl["per_bed"].items(), key=lambda kv: kv[1]["n_turns"])
            ref_sqft = mix.loc[mix["bed"] == ref_bed, "avg_sqft"]
            scale = (float(r["avg_sqft"]) / float(ref_sqft.iloc[0])
                     if len(ref_sqft) and pd.notna(r.get("avg_sqft"))
                     and pd.notna(ref_sqft.iloc[0]) and float(ref_sqft.iloc[0]) > 0 else 1.0)
            mk = round(ref["median_direct"] * scale, 0) + gl["pool_per_turn"]
            mk_method = (f"measured-scaled: {ref_bed}BR median x sqft ratio + pool "
                         f"(this bedroom too thin, n={n})")
        else:
            mk = FALLBACK_MAKE_READY
            mk_method = (f"FALLBACK flat ${FALLBACK_MAKE_READY:.0f} — GL U= coverage "
                         f"{gl['coverage']:.0%} below {_COVERAGE_MIN:.0%}; unmeasured")

        gross = net = rate = exp_loss = annual = None
        if vac_mo is not None and rent:
            gross = round(vac_mo * rent + mk, 0)
            net_pct = (relet - ri_pct) if (relet is not None and ri_pct is not None) else 0.0
            net = round(max(gross - net_pct / 100.0 * rent * _RELET_NET_TERM_MONTHS, 0.0), 0)
        oy = occ.get(bed)
        if oy and n:
            rate = round(n / oy, 3)
            if net is not None:
                exp_loss = round(rate * net, 0)
        if net is not None and n:
            annual = round(n / WINDOW_YEARS * net, 0)
        rows.append({
            "bedrooms": bed, "units": int(r["units"]),
            "active_residents": int(r["active_residents"] or 0),
            "avg_inplace_rent": round(rent, 0) if rent else None,
            "avg_market_rent": (round(float(r["market_rent"]), 0)
                                if pd.notna(r.get("market_rent")) else None),
            "turns_window": n,
            "vacancy_days_raw": round(raw_days, 1) if raw_days is not None else None,
            "vacancy_days_shrunk": round(shrunk, 1) if shrunk is not None else None,
            "vacancy_months": vac_mo,
            "make_ready": mk, "make_ready_method": mk_method,
            "relet_delta_pct": round(relet, 2) if relet is not None else None,
            "turnover_cost_gross": gross, "turnover_cost_net": net,
            "occupied_unit_years": round(oy, 1) if oy else None,
            "turnover_rate_per_unit_yr": rate,
            "expected_loss_per_unit_yr": exp_loss,
            "annual_turn_cost": annual})

    rows.sort(key=lambda x: x["bedrooms"])
    total = sum(r["annual_turn_cost"] or 0 for r in rows)
    return {"property_id": pid, "property": data.PROPERTIES.get(pid),
            "renewal_increase_pct": ri_pct,
            "make_ready_source": {"gl_coverage": gl["coverage"],
                                  "measured": bool(measured),
                                  "pool_per_turn": gl["pool_per_turn"],
                                  "gl_lines_attributed": gl["n_costs"]},
            "bedrooms": rows,
            "annual_turn_cost_total": round(total, 0)}


def build(property_ids: list[int] | None = None, as_of: str | None = None) -> dict:
    """The unit-economics ledger for one or more properties. `as_of` pins the end of the
    4-year window (turns, occupancy, renewal increases, GL) so a snapshot is reproducible;
    default today. Unit mix and in-place rent are always current state. Cached 24h."""
    pids = sorted(int(p) for p in (property_ids or data.DEFAULT_PROPERTY_IDS))
    asof = data._safe_asof(as_of)
    key = (tuple(pids), asof)
    hit = _CACHE.get(key)
    if hit and time.time() - hit[0] < _TTL:
        return hit[1]
    with _LOCK:
        hit = _CACHE.get(key)
        if hit and time.time() - hit[0] < _TTL:
            return hit[1]
        props = [_property_block(p, asof) for p in pids]
        out = {
            "contract": "multifamily-churn", "view": "unit-economics",
            "as_of": asof, "window_years": WINDOW_YEARS,
            "methods": {
                "turn": "move-out -> next move-in on the same unit, gap 0-365d, occupancy "
                        "statuses only (TENANT.ISTATUS 0/1/2/3/4; canceled/denied applications "
                        "excluded)",
                "window": f"{WINDOW_YEARS}y ending as_of={asof}; unit mix and in-place rent are "
                          "current state regardless of as_of",
                "vacancy_shrinkage": f"empirical Bayes toward property mean, k={SHRINK_K:.0f}",
                "denominator": "occupied unit-years as per-unit INTERVAL UNION of "
                               "tenancies (sum-of-spans double-counts co-tenancies)",
                "rent_basis": "in-place avg rent (active leases) for vacancy loss; "
                              "market rent reported alongside",
                "make_ready": "GL U= unit-note attribution (IBOOK=1, direct accounts "
                              "median + pool/turn) where coverage >= "
                              f"{_COVERAGE_MIN:.0%}; flat fallback otherwise. EXCLUDES "
                              "in-house maintenance labor (no make-ready split exists)",
                "relet_netting": "same-unit relet rent delta NET of the Lease_History "
                                 f"renewal increase, over {_RELET_NET_TERM_MONTHS} months",
                "no_assumptions": "no uplift, flip rate, or model score is used anywhere "
                                  "in this view"},
            "properties": props,
            "portfolio_annual_turn_cost": round(
                sum(p["annual_turn_cost_total"] for p in props), 0)}
        _CACHE[key] = (time.time(), out)
        return out
