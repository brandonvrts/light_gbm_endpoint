"""Ops-lever baseline audit (Phase 0 §11.4 follow-through, 2026-08-20).

Grounds the two deterministic retention levers before any concession spend:

1. VACANCY TAIL — the long turns (gap >= property p90) that drive mean vacancy above
   the median. For properties with GL U= unit-tagging (62W), each tail turn is
   classified by when make-ready work happened inside the window:
     ready_then_sat  work finished early, unit sat unrented   -> leasing problem
     late_work       work landed late in the window           -> make-ready problem
     heavy_reno      outsized spend (>= $2.5k)                -> capex hold, don't
                                                                 bill to churn
     no_gl_work      no attributed work at all                -> hold/unknown
2. SEASONAL FILL — median/mean days-to-fill by move-out season, per property (the
   physics behind lease-end migration).
3. LEASE-END EXPOSURE — current active executed terms (Lease_History iStatus 4)
   ending in the slow-fill window (Nov-Feb), per property.

Read-only; prints a report. Reproduces the numbers cited in the research baseline.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from core import db  # noqa: E402
from contracts.multifamily_churn import data, unit_economics as ue  # noqa: E402

HEAVY_RENO = 2500.0
WINTER = (11, 12, 1, 2)


def _tail(pid: int) -> None:
    turns = ue._turns_frame(pid)
    if turns.empty:
        print(f"  no turns in window")
        return
    p90 = float(turns["gap"].quantile(0.90))
    tail = turns[turns["gap"] >= p90].copy()
    daily = db.query(f"""
        SELECT u.IBEDROOMS bed, AVG(CASE WHEN t.ISTATUS=0 AND t.SRENT>200
                                         THEN CAST(t.SRENT AS FLOAT) END)/30.44 d
        FROM [{db.DB_YARDI}].dbo.UNIT u
        LEFT JOIN [{db.DB_YARDI}].dbo.TENANT t ON t.HUNIT=u.HMY
        WHERE u.HPROPERTY=? AND u.EXCLUDE=0 GROUP BY u.IBEDROOMS""", (pid,))
    drate = dict(zip(daily["bed"], pd.to_numeric(daily["d"], errors="coerce")))
    med = float(turns["gap"].median())
    tail["excess_days"] = tail["gap"] - med
    tail["excess_cost"] = tail.apply(
        lambda r: r["excess_days"] * float(drate.get(r["bed"]) or 0), axis=1)
    print(f"  gaps: median {med:.0f}d, mean {turns['gap'].mean():.0f}d, p90 {p90:.0f}d"
          f" | tail turns (>=p90): {len(tail)} of {len(turns)}"
          f" | excess vs median: {tail['excess_days'].sum():,.0f} days"
          f" ≈ ${tail['excess_cost'].sum():,.0f} over {ue.WINDOW_YEARS}y"
          f" (${tail['excess_cost'].sum()/ue.WINDOW_YEARS:,.0f}/yr)")

    # classify via GL where the U= convention holds
    gl = ue._gl_make_ready(pid, turns)
    if gl["coverage"] < ue._COVERAGE_MIN:
        print(f"  (no GL unit-tagging at this property — coverage {gl['coverage']:.0%};"
              f" tail classification needs a PM walk-through instead)")
        return
    accts = ",".join(f"'{a}'" for a in ue.DIRECT_ACCTS)
    lines = db.query(f"""
        SELECT g.DTDATE, g.DAMOUNT, g.SNOTES
        FROM [{db.DB_YARDI}].dbo.GLDETAIL g
        JOIN [{db.DB_YARDI}].dbo.ACCT a ON a.HMY=g.HACCT
        WHERE g.HPROP=? AND g.IBOOK=1 AND g.SNOTES LIKE 'U=%'
          AND g.DTDATE >= DATEADD(YEAR,-{ue.WINDOW_YEARS + 1},GETDATE())
          AND LTRIM(RTRIM(a.SCODE)) IN ({accts})""", (pid,))
    lines = lines.assign(
        ucode=lines["SNOTES"].astype(str).str.extract(ue._UNIT_NOTE, expand=False)
              .str.strip().str.lower(),
        DTDATE=pd.to_datetime(lines["DTDATE"]),
        DAMOUNT=pd.to_numeric(lines["DAMOUNT"], errors="coerce")).dropna(subset=["ucode"])
    kinds: dict[str, list] = {}
    for r in tail.to_dict("records"):
        w = lines[(lines["ucode"] == r["ucode"]) &
                  (lines["DTDATE"] >= r["DTMOVEOUT"] - pd.Timedelta(days=7)) &
                  (lines["DTDATE"] <= r["next_in"] + pd.Timedelta(days=45))]
        if w.empty:
            kind = "no_gl_work"
        elif float(w["DAMOUNT"].sum()) >= HEAVY_RENO:
            kind = "heavy_reno"
        else:
            # position of the LAST work inside the vacancy (0=at move-out, 1=at move-in)
            span = max((r["next_in"] - r["DTMOVEOUT"]).days, 1)
            pos = (w["DTDATE"].max() - r["DTMOVEOUT"]).days / span
            kind = "ready_then_sat" if pos <= 0.5 else "late_work"
        kinds.setdefault(kind, []).append(r)
    for kind in ("ready_then_sat", "late_work", "heavy_reno", "no_gl_work"):
        rows = kinds.get(kind, [])
        if not rows:
            continue
        xd = sum(x["excess_days"] for x in rows)
        xc = sum(x["excess_cost"] for x in rows)
        print(f"    {kind:15s} n={len(rows):3d}  excess {xd:5,.0f}d  ≈ ${xc:9,.0f}"
              + ("  <- leasing lever" if kind == "ready_then_sat" else
                 "  <- exclude from churn billing" if kind == "heavy_reno" else ""))
        worst = sorted(rows, key=lambda x: -x["excess_days"])[:3]
        for x in worst:
            print(f"        unit {x['ucode']:8s} {x['bed']}BR  out {x['DTMOVEOUT']:%Y-%m-%d}"
                  f"  gap {x['gap']:.0f}d")


def _seasonal(pid: int) -> None:
    turns = ue._turns_frame(pid)
    if turns.empty:
        return
    t = turns.assign(m=turns["DTMOVEOUT"].dt.month)
    t["season"] = t["m"].map(lambda m: "Nov-Feb" if m in WINTER
                             else ("Mar-Jun" if m in (3, 4, 5, 6) else "Jul-Oct"))
    g = t.groupby("season")["gap"].agg(["count", "mean", "median"]).round(1)
    print("  fill by move-out season:", {i: f"n={int(r['count'])} med={r['median']:.0f}d "
          f"mean={r['mean']:.0f}d" for i, r in g.iterrows()})
    if {"Nov-Feb", "Mar-Jun"} <= set(g.index):
        win, spr = g.loc["Nov-Feb"], g.loc["Mar-Jun"]
        n_win_yr = win["count"] / ue.WINDOW_YEARS
        print(f"  winter penalty: {win['mean'] - spr['mean']:+.0f} mean days/turn"
              f" x {n_win_yr:.1f} winter move-outs/yr")


def _lease_end_exposure(pid: int) -> None:
    df = db.query(f"""
        SELECT MONTH(lh.dtLeaseTo) m, COUNT(*) n
        FROM [{db.DB_YARDI}].dbo.Lease_History lh
        JOIN [{db.DB_YARDI}].dbo.TENANT t ON t.HMYPERSON = lh.hTent
        JOIN [{db.DB_YARDI}].dbo.UNIT u ON u.HMY = t.HUNIT
        WHERE t.HPROPERTY = ? AND u.EXCLUDE = 0 AND lh.iStatus = 4
          AND t.ISTATUS = 0 AND t.DTMOVEOUT IS NULL AND lh.dtLeaseTo >= GETDATE()
        GROUP BY MONTH(lh.dtLeaseTo) ORDER BY m""", (pid,))
    if df.empty:
        print("  no active future lease-ends found")
        return
    total = int(df["n"].sum())
    winter = int(df[df["m"].isin(WINTER)]["n"].sum())
    dist = {int(r["m"]): int(r["n"]) for r in df.to_dict("records")}
    print(f"  active future lease-ends: {total} | ending Nov-Feb: {winter}"
          f" ({100*winter/total:.0f}%) | by month: {dist}")


if __name__ == "__main__":
    pids = ([int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1
            else list(data.DEFAULT_PROPERTY_IDS))
    for pid in pids:
        print(f"\n=== {data.PROPERTIES.get(pid, pid)} ({pid}) ===")
        print("-- vacancy tail --")
        _tail(pid)
        print("-- seasonality --")
        _seasonal(pid)
        print("-- lease-end exposure --")
        _lease_end_exposure(pid)
