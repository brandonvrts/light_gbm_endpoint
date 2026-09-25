"""Smoke + golden regression tests for the multifamily churn contract.

Two layers:
  * OFFLINE (always run): schema/bundle invariants + calibrator monotonicity. These catch
    version/arch skew and feature-count drift with no DB.
  * DB-GATED (skip cleanly if SSMS is unreachable): behavioral goldens that pin the fixes
    from the roadmap -- Kathleen Gray reads low, the April 2026 cohort reconstructs, tiers
    are sane, and v6 scores are not snapped.

Run: PYTHONPATH=. .venv/bin/python tests/test_smoke.py   (or via pytest if available)
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class Skip(Exception):
    pass


def _db_ok() -> bool:
    try:
        from core import db
        return db.healthcheck().get("db") == "ok"
    except Exception:
        return False


# ---------------------------------------------------------------- OFFLINE
def test_schema_feature_count_matches_model_features():
    from contracts.multifamily_churn import schema, data
    assert len(data.MODEL_FEATURES) == schema.EXPECTED_FEATURE_COUNT == 46
    assert len(set(data.MODEL_FEATURES)) == len(data.MODEL_FEATURES), "duplicate feature"
    assert schema.VARIANTS == ["h60", "h180"]


def test_deployed_bundle_loads_and_matches_manifest():
    from core import registry
    b = registry.load("multifamily_churn")
    fcols = b["feature_columns"]
    assert len(fcols) in (45, 46), f"unexpected feature count {len(fcols)}"
    man = registry.manifest("multifamily_churn")
    if man.get("feature_columns"):
        assert list(man["feature_columns"]) == list(fcols), "manifest/bundle feature skew"
    # tier thresholds must be a sane 0 < low < med < 1
    th = b.get("thresholds") or {}
    lo, mid = th.get("risk_low_max"), th.get("risk_medium_max")
    assert 0 < lo < mid < 1, f"degenerate tiers low={lo} med={mid}"


def test_spline_calibrator_is_monotone_and_smooth():
    import numpy as np
    from calibration import SplineCalibrator

    class _Stub:                       # raw score = feature 0 passed through a sigmoid
        def predict_proba(self, X):
            p = 1 / (1 + np.exp(-np.asarray(X)[:, 0]))
            return np.column_stack([1 - p, p])

    rng = np.random.RandomState(0)
    z = rng.normal(0, 2, 4000)
    y = (rng.uniform(size=4000) < 1 / (1 + np.exp(-(z - 1)))).astype(int)
    X = z.reshape(-1, 1)
    cal = SplineCalibrator.fit(_Stub(), X, y, floor=0.005, ceil=0.95)
    grid = np.linspace(-6, 6, 200).reshape(-1, 1)
    p = cal.predict_proba(grid)[:, 1]
    assert np.all(np.diff(p) >= -1e-9), "calibrator not monotone"
    assert len(set(np.round(p, 4))) > 50, "calibrator output looks snapped"
    assert p.min() >= 0.005 - 1e-9 and p.max() <= 0.95 + 1e-9


def test_memo_categories_precise():
    from contracts.multifamily_churn import data
    # regression: the bare %notice% / %emergency% false-positive buckets are gone
    assert "notice" not in data.MEMO_CATEGORIES["EvictionMemos"]
    joined = " ".join(data.MEMO_CATEGORIES["MaintenanceMemos"])
    assert "emergency" not in joined


# ---------------------------------------------------------------- DB-GATED
def test_kathleen_gray_reads_low():
    if not _db_ok():
        raise Skip("no DB")
    from contracts.multifamily_churn import contract as mf
    t = (mf.lookup_tenant("Kathleen Gray") or {}).get("tenant") or {}
    assert t.get("found"), "Kathleen Gray not resolved"
    assert (t.get("tenure_months") or 0) > 150, f"tenure not person-level: {t.get('tenure_months')}"
    assert t["churn_probability"] < 0.15, f"14y no-notice resident scored high: {t['churn_probability']}"
    assert t["risk_tier"] == "low"
    assert t.get("why"), "missing explanation"


def test_april_2026_cohort_reconstructs():
    if not _db_ok():
        raise Skip("no DB")
    from contracts.multifamily_churn import data
    n = len(data.scodes_by_lease_window("2026-04-01", "2026-04-30"))
    assert n >= 70, f"April cohort collapsed to {n} (Lease_History reconstruction broken)"


def test_scores_not_snapped():
    if not _db_ok():
        raise Skip("no DB")
    from contracts.multifamily_churn import contract as mf
    rep = mf.monthly_churn_report("2026-09")
    scored = [r["churn_probability"] for r in rep["results"]
              if r.get("churn_probability") is not None]
    if len(scored) < 20:
        raise Skip("cohort too small to judge snapping")
    distinct = len(set(scored))
    assert distinct >= 0.5 * len(scored), f"scores look snapped: {distinct}/{len(scored)} distinct"


# ---------------------------------------------------------------- BROKERAGE (Phase 0.5)
def test_brokerage_registry_guard_blocks_shadow():
    """Tombstoned/shadow families must never resolve for serving; brokerage serves v0.5."""
    from core import registry
    assert registry.default_version("signal_desk_dealwin") is None, "dealwin must not serve"
    assert registry.default_version("signal_desk_recency") is None, "recency must not serve"
    bk = registry.default_version("signal_desk_brokerage")
    assert bk and bk.startswith("v0.5"), f"brokerage default should be v0.5, got {bk}"
    # introspection still reaches the tombstoned bundle (for /models cards)
    assert registry.latest_version("signal_desk_dealwin") is not None


def test_brokerage_split_bundle_shape_and_tiers():
    from core import registry
    b = registry.load("signal_desk_brokerage")
    assert b.get("bundle_format") == "brokerage-split-v0.5"
    assert set(b["models"]) == {"pipeline_momentum", "cold_fit", "combined"}
    pol = (b.get("opportunity_tiers") or {}).get("combined") or {}
    assert pol.get("hot_min", 0) > pol.get("warm_min", -1), "tier cutoffs not ordered"
    # honest metric present and modest (small-n, must not read like a strong model)
    m = b["metrics"]["head_combined"]
    assert 0.0 < m["pr_auc"] < 0.5 and m["pr_auc"] > m["pr_auc_no_skill"]


def test_brokerage_scorecard_default_sort_ok():
    if not _db_ok():
        raise Skip("no DB")
    from contracts.signal_desk_brokerage import scorecard
    for s in ("score", "winprob", "value", "lease"):
        out = scorecard.company_scorecard(s, 5, None)   # default sort used to 500
        assert out["count"] >= 0 and "companies" in out, f"sort={s} failed"


def test_brokerage_freshness_flag():
    if not _db_ok():
        raise Skip("no DB")
    from contracts.signal_desk_brokerage import data as bkd
    f = bkd.feature_freshness()
    assert "stale" in f and isinstance(f["stale"], bool)


def test_brokerage_score_company_heads():
    if not _db_ok():
        raise Skip("no DB")
    from contracts.signal_desk_brokerage import contract as bk
    r = bk.score_company("5", include_history=False)
    if not r.get("history"):
        raise Skip("company 5 has no feature history")
    row = r["history"][-1]
    for k in ("conversion_probability", "pipeline_momentum", "cold_fit", "opportunity_tier"):
        assert k in row, f"missing {k} in split-bundle history row"
    assert "data_stale" in r


def test_spacedemand_pipeline_dedup_fires():
    """Regression guard: the in_pipeline dedup silently no-op'd once (wrong column name
    swallowed by a bare except). The open-deal set must be non-empty."""
    if not _db_ok():
        raise Skip("no DB")
    from contracts.signal_desk_spacedemand import data as sd
    assert len(sd._pipeline_company_ids()) > 0, "in_pipeline dedup resolves an empty set (query broken?)"


# ------------------------------------------------- MULTIFAMILY COHORTS (chatbot)
def test_mf_cohort_registry_shape():
    """Every cohort carries label/description/definition/predicate; the survivor-biased
    rent-hike cohorts and the non-differentiating bare work-order cohort stay excluded."""
    from contracts.multifamily_churn import cohorts as mc
    assert len(mc.COHORTS) == 10
    for k, c in mc.COHORTS.items():
        assert k == k.lower() and " " not in k
        for f in ("label", "description", "definition", "predicate"):
            assert c.get(f), f"{k} missing {f}"
    assert not any("rent" in k for k in mc.COHORTS), "rent-hike cohorts are survivor-biased"
    assert "recent-work-order" not in mc.COHORTS, "bare work-order cohort is non-differentiating"


def test_mf_cohort_catalog_and_report():
    """Catalog + report agree with each other and with the validated analysis: holdover
    is the standout (lift > 2), pending-decision excludes notice-givers, math is coherent."""
    if not _db_ok():
        raise Skip("no DB")
    from contracts.multifamily_churn import cohorts as mc
    cat = mc.catalog()
    assert cat["population"]["n"] > 800, f"population collapsed: {cat['population']}"
    assert 0.02 < cat["population"]["avg_churn"] < 0.30
    keys = {c["cohort"] for c in cat["cohorts"]}
    assert keys == set(mc.COHORTS)
    hold = next(c for c in cat["cohorts"] if c["cohort"] == "holdover-mtm")
    assert hold["n"] > 20 and hold["lift_vs_portfolio"] > 2.0, f"holdover lost its signal: {hold}"

    rep = mc.report("holdover-mtm", top=5)
    assert rep["n"] == hold["n"] and rep["expected_moveouts"] == hold["expected_moveouts"]
    assert abs(rep["avg_churn"] * rep["n"] - rep["expected_moveouts"]) < 1.0, "sum/mean mismatch"
    assert sum(rep["tier_mix"].values()) == rep["n"]
    assert len(rep["members_top"]) == 5
    probs = [m["churn_probability"] for m in rep["members_top"]]
    assert probs == sorted(probs, reverse=True), "members not ranked by risk"
    assert rep["summary"] and str(rep["n"]) in rep["summary"], "summary not chatbot-ready"

    pend = mc.report("pending-decision", top=1)
    assert pend["notice_given_n"] == 0, "pending-decision must exclude notice-givers"
    try:
        mc.report("no-such-cohort")
        assert False, "unknown cohort should raise"
    except ValueError as e:
        assert "holdover-mtm" in str(e), "error should list valid cohorts"


# ------------------------------------------------- PERSUADABLES (incentive targeting)
def test_persuadables_band_and_economics():
    """The persuadable list must be band-pure (0.2-0.6, zero notice), explain every
    member via cohort memberships + model drivers, and keep scenario math coherent."""
    if not _db_ok():
        raise Skip("no DB")
    from contracts.multifamily_churn import persuadables as pers
    out = pers.build(top=100)
    assert out["population"]["n"] > 800
    assert out["n"] > 10, f"persuadable band suspiciously empty ({out['n']})"
    assert out["n"] == len(out["members"]) or len(out["members"]) == 100
    for m in out["members"]:
        assert 0.20 <= m["churn_probability"] < 0.60, f"out-of-band member {m['scode']}"
        fence = m["on_the_fence_because"]
        assert "risk_cohorts" in fence and "protective_cohorts" in fence
        # value_if_saved = turn cost NET of the re-lease premium — non-negative, and
        # (given our measured premiums) below the raw vacancy+make-ready ceiling
        assert 0 <= m["value_if_saved"] < 10000, f"save value implausible: {m['value_if_saved']}"
        assert m["monthly_rent_at_risk"] <= (m["monthly_rent"] or 0) + 1e-6 or m["monthly_rent"] is None
    # ranked by leverage (rent at risk)
    rar = [m["monthly_rent_at_risk"] for m in out["members"]]
    assert rar == sorted(rar, reverse=True)
    # at least one member shows a genuine conflict (both risk and protective cohorts)
    assert any(m["on_the_fence_because"]["risk_cohorts"] and
               m["on_the_fence_because"]["protective_cohorts"] for m in out["members"]), \
        "no member shows the risk+protective conflict the surface exists to explain"
    # model-side why present for most members (explanations are the PM value-add)
    with_why = sum(1 for m in out["members"] if m["why"])
    assert with_why >= 0.8 * len(out["members"]), f"explanations missing ({with_why}/{len(out['members'])})"
    # the band is scored on the h180 renewal forecast, never the h60 triage default
    assert out["model_version"] == pers.RENEWAL_BUNDLE, out["model_version"]
    assert out["horizon_days"] == 180
    # DEFAULT MODE (no flip_rate): assumption-free breakeven, no projected saves
    sc0 = out["economics"]["scenario"]
    assert sc0["assumed_flip_rate"] is None and sc0["expected_saves"] is None \
        and sc0["pays"] is None
    assert "NONE PASSED" in sc0["flip_rate_note"]
    # breakeven flip rate is where spend == value saved
    assert abs(sc0["breakeven_flip_rate"] * out["expected_moveouts"] * sc0["value_per_save"]
               - sc0["total_spend"]) < 0.02 * sc0["total_spend"] + 50
    # SCENARIO MODE (explicit flip_rate): the projected-save math must stay coherent
    out = pers.build(top=100, flip_rate=0.25)
    sc = out["economics"]["scenario"]
    assert sc["offers"] == out["n"]
    assert abs(sc["total_spend"] - sc["offers"] * sc["incentive_cost_per_offer"]) < 1.0
    assert abs(sc["expected_saves"] - out["expected_moveouts"] * sc["assumed_flip_rate"]) < 0.11
    assert "ASSUMED" in sc["flip_rate_note"]
    # cost_per_save = spend / saves; `pays` iff it undercuts value_per_save; breakeven
    # concession is where spend == value (expected_saves is display-rounded -> loose tol)
    assert abs(sc["cost_per_save"] - sc["total_spend"] / sc["expected_saves"]) \
        < 0.05 * sc["cost_per_save"] + 5
    assert sc["pays"] == (sc["cost_per_save"] <= sc["value_per_save"])
    # both sides reconstructed from display-rounded fields -> relative tolerance
    assert abs(sc["breakeven_concession_per_offer"] * sc["offers"]
               - sc["expected_saves"] * sc["value_per_save"]) \
        < 0.03 * sc["expected_saves"] * sc["value_per_save"] + 50
    # do-nothing baseline: turnover loss consistent with expected move-outs x values
    dn = out["economics"]["do_nothing"]
    assert dn["expected_moveouts"] == out["expected_moveouts"]
    assert dn["turnover_loss"] > 0
    # re-lease premium measured (all 5 properties re-let at a premium as of 2026-07)
    rp = out["economics"]["relet_premium"]
    assert rp["per_property"], "relet premium query returned nothing"
    assert rp["portfolio_avg_pct"] > 0, "portfolio relet premium expected positive"
    assert rp["renewal_increase_pct_by_property"], "renewal increases missing"
    # turn-days measured (not the fallback) for at least one property
    td = out["economics"]["turn_days"]
    assert td["per_property"], "turn days query returned nothing (fallback everywhere)"
    # economics must be invariant to how many members are displayed (top-N slice bug)
    sc1 = pers.build(top=1, flip_rate=0.25)["economics"]["scenario"]
    assert sc1["expected_value_saved"] == sc["expected_value_saved"], \
        "scenario economics vary with `top` (computed over displayed slice, not the band)"
    # band parameters propagate + junk rejected
    try:
        pers.build(band_low=0.7, band_high=0.3)
        assert False, "inverted band should raise"
    except ValueError:
        pass


def test_outreach_incentive_actions():
    from contracts.multifamily_churn import outreach
    assert {"incentive_offered", "incentive_accepted", "incentive_declined"} <= outreach._ACTIONS


def test_outreach_typed_offers():
    """incentive_* events REQUIRE a typed offer (the per-offer flip-rate denominator);
    junk types/structures/arms are rejected; a full typed offer round-trips."""
    import tempfile
    from pathlib import Path
    from contracts.multifamily_churn import outreach
    orig = outreach._STORE
    try:
        outreach._STORE = Path(tempfile.mkdtemp()) / "outreach_test.jsonl"
        for bad_kwargs in (dict(),                                   # missing offer_type
                           dict(offer_type="one_month_off"),         # unknown type
                           dict(offer_type="free_months_1", offer_structure="magic"),
                           dict(offer_type="free_months_1", arm="placebo")):
            try:
                outreach.record("t0000001", "incentive_offered", **bad_kwargs)
                assert False, f"should have raised: {bad_kwargs}"
            except ValueError:
                pass
        rec = outreach.record("t0000001", "incentive_offered",
                              offer_type="free_months_1",
                              offer_structure="one_time_credit", offer_value=1.0,
                              incentive_cost=1668.0, property_id=82, bedroom=2,
                              arm="treated", posting_account=53)
        assert rec["offer_type"] == "free_months_1" and rec["arm"] == "treated"
        assert rec["posting_account"] == 53
        # non-incentive actions stay permissive (no typed offer needed)
        assert outreach.record("t0000001", "called")["action"] == "called"
        evs = outreach._read()
        assert len(evs) == 2 and evs[0]["offer_type"] == "free_months_1"
    finally:
        outreach._STORE = orig


def test_unit_economics_ledger():
    """The Phase 1 decision ledger: per-bedroom turnover economics must be
    assumption-free, method-labeled, and internally coherent."""
    if not _db_ok():
        raise Skip("no DB")
    from contracts.multifamily_churn import unit_economics as ue
    out = ue.build([82])
    assert out["view"] == "unit-economics" and len(out["properties"]) == 1
    p = out["properties"][0]
    # 62W is the property whose GL carries the U= convention — must be measured
    assert p["make_ready_source"]["measured"] is True
    assert p["make_ready_source"]["gl_coverage"] > 0.9
    beds = {b["bedrooms"]: b for b in p["bedrooms"]}
    assert set(beds) >= {1, 2}, f"bedroom classes missing: {sorted(beds)}"
    for b in p["bedrooms"]:
        if b["turnover_cost_gross"] is None:
            continue
        # gross = vacancy months x in-place rent + make-ready (rounding tolerance)
        recon = b["vacancy_months"] * b["avg_inplace_rent"] + b["make_ready"]
        assert abs(b["turnover_cost_gross"] - recon) < 2, (b["bedrooms"], recon)
        # annual = turns/window x net
        if b["annual_turn_cost"] is not None:
            assert abs(b["annual_turn_cost"]
                       - b["turns_window"] / out["window_years"] * b["turnover_cost_net"]) < 2
        # rate uses occupied unit-years, which can never exceed units x window
        if b["occupied_unit_years"] is not None:
            assert b["occupied_unit_years"] <= b["units"] * out["window_years"] + 0.1
        assert b["make_ready_method"]
    assert p["annual_turn_cost_total"] > 0
    assert "no_assumptions" in out["methods"]


def test_incentive_accounts_registry():
    """The 8 landlord concession accounts, and never the subsidy account (40 = income)."""
    from contracts.multifamily_churn import incentives
    assert len(incentives.ACCOUNTS) == 8
    assert 53 in incentives.ACCOUNTS and incentives.ACCOUNTS[53] == "Renewal Concession"
    assert 40 not in incentives.ACCOUNTS, "subsidy is income, not a concession"


def test_edge_residents_month_filter_and_incentives():
    """'Edge residents for <month>' must scope every member to that decision month,
    and each member must carry incentive history (the surfaceable-in-API requirement)."""
    if not _db_ok():
        raise Skip("no DB")
    from contracts.multifamily_churn import persuadables as pers, incentives
    from datetime import date
    all_band = pers.build(top=100)
    assert all_band["n"] > 0
    # every member surfaces an incentives block with the full shape
    for m in all_band["members"]:
        inc = m["incentives"]
        assert inc is not None and {"past_concessions_n", "past_concessions_total",
                                    "recent_concessions", "offers_logged_n"} <= set(inc)
    # pick the most common decision month among members and filter to it
    months = {}
    for m in all_band["members"]:
        if m["lease_to"]:
            months[m["lease_to"][:7]] = months.get(m["lease_to"][:7], 0) + 1
    target = max(months, key=months.get)
    scoped = pers.build(top=100, month=target)
    assert 0 < scoped["n"] <= all_band["n"]
    assert all((m["lease_to"] or "")[:7] == target for m in scoped["members"]), \
        f"member outside month {target}"
    assert target in scoped["summary"]
    try:
        pers.build(month="August")
        assert False, "non-YYYY-MM month should raise"
    except ValueError:
        pass
    # incentive history module reads real TRANS rows (any concession-taker will do),
    # and totals cover the FULL filtered set — independent of the display limit
    hist = incentives.history(property_id=62, limit=5)
    assert hist["totals"]["events"] > len(hist["events"]), \
        "totals look capped by `limit` (CCV has decades of concessions)"
    assert hist["events_returned"] == len(hist["events"]) <= 5
    assert all(e["account_name"] != "unknown" for e in hist["events"])
    assert hist["totals"]["events"] == incentives.history(property_id=62, limit=1)["totals"]["events"]


# ------------------------------------------------- EDGE LIST (hysteresis)
def test_edge_hysteresis_rules_offline():
    """The pure decision function: enter at 0.20, survive dips to 0.15, exit only on
    resolution (renewal/notice/move-out) or p<0.15; fight_for sticky from entry."""
    from contracts.multifamily_churn.edge import apply_hysteresis

    def row(scode, p, notice=False, lease_to="2026-09-01", trait=5.0, pct=20.0):
        return {"scode": scode, "name": scode, "unit": "1", "property_id": 62, "p": p,
                "notice": notice, "lease_to": lease_to, "d2e": 30, "trait": trait,
                "trait_pctile": pct, "monthly_rent": 1000.0}

    # cycle 1: two entrants; higher-trait one becomes fight_for
    rows = {"a": row("a", 0.30, trait=8.0), "b": row("b", 0.25, trait=2.0),
            "c": row("c", 0.18)}                       # c: watch band, must NOT enter
    st, ev = apply_hysteresis({}, rows, "2026-07-01")
    assert set(st) == {"a", "b"} and st["a"]["fight_for"] and not st["b"]["fight_for"]
    assert all(e["event"] == "enter" for e in ev)

    # cycle 2: a dips to 0.17 (flap zone) -> STAYS, fight_for unchanged; b renews -> exits
    rows2 = {"a": row("a", 0.17, trait=1.0), "b": row("b", 0.25, lease_to="2027-09-01")}
    st2, ev2 = apply_hysteresis(st, rows2, "2026-08-01")
    assert "a" in st2 and st2["a"]["fight_for"] is True, "dip to 0.17 must not evict"
    assert st2["a"]["trait_at_entry"] == 8.0, "entry snapshot must be preserved"
    assert "b" not in st2 and any(e["reason"] == "renewed" for e in ev2)

    # cycle 3: a below 0.15 -> exits; d gives notice while listed -> exits; e moved out
    st2["d"] = {**st2.get("d", {}), "scode": "d", "entered": "2026-08-01",
                "p_at_entry": 0.3, "trait_at_entry": 3.0, "lease_to_at_entry": "2026-09-01",
                "fight_for": False}
    st2["e"] = {"scode": "e", "entered": "2026-08-01", "p_at_entry": 0.3,
                "trait_at_entry": 3.0, "lease_to_at_entry": "2026-09-01", "fight_for": False}
    rows3 = {"a": row("a", 0.10), "d": row("d", 0.40, notice=True)}
    st3, ev3 = apply_hysteresis(st2, rows3, "2026-09-01")
    reasons = {e["scode"]: e["reason"] for e in ev3 if e["event"] == "exit"}
    assert st3 == {} and reasons == {"a": "risk_subsided", "d": "gave_notice", "e": "moved_out"}


def test_edge_list_live_idempotent():
    """Live: same-day calls return identical membership; invariants hold; historical
    as_of never rewrites the stored roster."""
    if not _db_ok():
        raise Skip("no DB")
    from contracts.multifamily_churn import edge
    a = edge.edge_list()
    b = edge.edge_list()
    assert {m["scode"] for m in a["members"]} == {m["scode"] for m in b["members"]}, \
        "same-day edge list not idempotent"
    assert b["changes_this_refresh"] == [], "second same-day refresh must be a no-op"
    for m in a["members"]:
        assert m["p_at_entry"] >= edge.ENTER_P - 1e-9
        assert m["churn_probability"] >= edge.EXIT_P - 1e-9, "member below exit line survived"
        assert isinstance(m["fight_for"], bool)
    for w in a["watch"]:
        assert edge.WATCH_P - 1e-9 <= w["churn_probability"] < edge.ENTER_P + 1e-9
    assert a["fight_for_n"] >= 1, "no fight-for residents at all (suspicious)"
    # preview of an old date must not move state
    before = edge._load_state()
    edge.edge_list(as_of="2026-01-01")
    assert edge._load_state() == before, "historical as_of rewrote the roster"


# ------------------------------------------------- MARKETING COHORTS (Phase 1)
def test_marketing_cohort_selector_parses():
    from contracts.marketing import space_demand as sdm
    assert sdm.parse_cohort("size=mid|vertical=healthcare") == ("mid", "healthcare")
    assert sdm.parse_cohort("vertical=healthcare") == (None, "healthcare")
    assert sdm.parse_cohort("healthcare") == (None, "healthcare")
    assert sdm.parse_cohort("mid") == ("mid", None)
    assert sdm.parse_cohort("global") == (None, None)
    for junk in ("size=huge", "fintech", "size=mid|vertical=nope"):
        try:
            sdm.parse_cohort(junk)
            assert False, f"'{junk}' should not parse"
        except ValueError:
            pass


def test_marketing_space_demand_rollup():
    """Cohort rollup must be an exact partition of the in-market scored universe, with
    calibrated sums as expected-event counts and a top-decile hot definition."""
    if not _db_ok():
        raise Skip("no DB")
    from contracts.marketing import space_demand as sdm
    out = sdm.cohort_rollup(market="IA,NE", min_scored=1)
    t = out["totals"]
    assert t["scored_in_market"] > 1500, f"in-market universe too small: {t}"
    assert out["model_source"]["contract"] == "signal-desk-spacedemand"
    assert out["model_source"]["version"], "model_source must carry the bundle version"
    # cohorts (min_scored=1) partition the universe exactly
    assert sum(c["n_scored"] for c in out["cohorts"]) == t["scored_in_market"]
    # hot = top decile -> ~10% of scored
    assert abs(t["hot_total"] - 0.10 * t["scored_in_market"]) <= 0.05 * t["scored_in_market"]
    # expected events: calibrated sum, positive, no cohort exceeds its own n
    assert 0 < t["expected_events_180d"] < t["scored_in_market"]
    for c in out["cohorts"]:
        assert c["expected_events_180d"] <= c["n_scored"] + 1e-6
        assert c["hot_new_n"] <= c["hot_n"] <= c["n_scored"]
        k = c["cohort_key"]
        assert k == "global" or k.startswith(("size=", "vertical=")), f"bad key {k}"
    # sorted by expected demand
    evs = [c["expected_events_180d"] for c in out["cohorts"]]
    assert evs == sorted(evs, reverse=True)
    # honest framing must be present in the response
    assert "NOT a conversion predictor" in out["note"]


def test_marketing_space_demand_leads_view():
    """In-cohort leads must be the same propensities the brokerage endpoint serves
    (one scoring pass), correctly filtered to the requested cohort."""
    if not _db_ok():
        raise Skip("no DB")
    from contracts.marketing import space_demand as sdm
    roll = sdm.cohort_rollup(market="IA,NE", min_scored=5)
    assert roll["cohorts"], "no cohorts with >=5 scored companies?"
    pick = roll["cohorts"][0]
    out = sdm.cohort_leads(pick["cohort_key"], market="IA,NE", limit=10)
    assert out["scored_in_cohort"] == pick["n_scored"]
    assert out["count"] == min(10, pick["n_scored"])
    props = [x["propensity"] for x in out["leads"]]
    assert props == sorted(props, reverse=True), "leads not ranked"
    for x in out["leads"]:
        assert x["cohort_key"] == pick["cohort_key"], "lead outside requested cohort"
    # cross-check the shared scoring pass: cohort top lead == that company's propensity
    # in the universe frame (identical, not merely close)
    df, _b = sdm.tagged_universe(None)
    cid = out["leads"][0]["company_id"]
    uni_p = round(float(df.loc[df["CompanyID"] == cid, "propensity"].iloc[0]), 4)
    assert out["leads"][0]["propensity"] == uni_p


def test_marketing_targeting_fused():
    """Phase 2: cohort_targeting must fuse the space-demand rollup (expected_opportunity =
    hot NEW leads x win-rate) while keeping every legacy field and honest no-data semantics."""
    if not _db_ok():
        raise Skip("no DB")
    from contracts.marketing import data as mkd
    out = mkd.cohort_targeting()
    assert out["space_demand_model"] and out["space_demand_model"]["version"], \
        "fusion missing model_source (rollup failed silently?)"
    assert out["cohorts"], "no targeting cohorts"
    fused = [c for c in out["cohorts"] if c["needs_space_now"]]
    assert fused, "no cohort matched the space-demand rollup (join key drift?)"
    for c in out["cohorts"]:
        # legacy fields must survive for the existing UI
        for k in ("vertical", "size_band", "addressable", "win_rate", "expected_winnable",
                  "median_days_in_stage"):
            assert k in c, f"legacy field {k} missing"
        assert c["cohort_key"].startswith("size="), "cohort rows must carry the full key"
        s = c["needs_space_now"]
        if s is None:
            assert c["expected_opportunity"] == 0.0, "no-data cohort must not fabricate opportunity"
        else:
            assert c["expected_opportunity"] == round(s["hot_new_n"] * c["win_rate"], 2)
            assert s["hot_new_n"] <= s["hot_n"] <= s["n_scored"]
    ranks = [(c["expected_opportunity"], c["expected_winnable"]) for c in out["cohorts"]]
    assert ranks == sorted(ranks, reverse=True), "cohorts not ranked by fused opportunity"
    # vertical rollup fused with ALL-sizes stats (not the unknown-size slice)
    v_fused = [v for v in out["verticals"] if v["needs_space_now"]]
    assert v_fused, "no vertical matched the rollup"
    assert "scoop_momentum_90d" in out and "no data, NOT no demand" in out["note"]


def test_marketing_options_never_500():
    from contracts.marketing import space_demand as sdm
    o = sdm.cohort_options()
    assert "size_bands" in o and "verticals" in o
    if _db_ok():
        assert o.get("cohort_keys"), "live options should list non-empty cohort keys"


# ------------------------------------------------- MARKETING COHORTS (Phase 0)
def test_cohort_vocab_offline():
    """The Python CohortVocab port must reproduce the C#/JS semantics exactly."""
    from contracts.marketing import cohort as cv
    # size thresholds are 25/100/500/2000 (NOT the spacedemand display bands)
    assert [cv.size_band_of(x) for x in (1, 24, 25, 99, 100, 499, 500, 1999, 2000)] == \
        ["micro", "micro", "small", "small", "mid", "mid", "large", "large", "enterprise"]
    assert cv.size_band_of(None) == cv.size_band_of(0) == cv.size_band_of(-5) == "unknown"
    # first-match-wins ordering: 'account' hits finance-insurance before professional-services
    assert cv.vertical_of("Accounting Services") == "finance-insurance"
    assert cv.vertical_of("Healthcare Services") == "healthcare"
    assert cv.vertical_of("Commercial & Residential Construction") == "real-estate"
    assert cv.vertical_of("Organizations") == "other"
    assert cv.vertical_of("") is None and cv.vertical_of(None) is None
    # key construction + backoff
    assert cv.cohort_key("mid", "healthcare") == "size=mid|vertical=healthcare"
    assert cv.cohort_key(None, None) == "global"
    assert cv.cohort_keys_for("mid", "unknown") == [(1, "size=mid"), (3, "global")]
    # range banding uses its own thresholds (20/100/500/5000, 'over 10k' -> enterprise)
    assert cv.band_from_employee_range("over 10,000") == "enterprise"
    assert cv.band_from_employee_range("1,000 - 5,000") == "large"
    assert cv.band_from_employee_range(None) == "unknown"


def test_cohort_key_parity_with_dealpattern():
    """Every COMPANY-grain rea.DealPattern row (BackoffLevel 0-3: size=/vertical=/global)
    must be reproduced by cohort_key(SizeBand, Vertical) — the join-key contract with the
    marketing app; divergence = silent join misses. The sizesqft= family (BackoffLevel 4,
    build-deal-patterns.mjs A4) is DEAL-sqft-grain, overloads the SizeBand column, and is
    intentionally excluded — it is not a company cohort dimension."""
    if not _db_ok():
        raise Skip("no DB")
    from core import db
    from contracts.marketing import cohort as cv
    df = db.query(f"SELECT CohortKey, SizeBand, Vertical FROM [{db.DB_MAIN}].rea.DealPattern")
    assert len(df) >= 50, f"DealPattern unexpectedly small ({len(df)})"
    company_grain = [r for r in df.to_dict("records")
                     if not str(r["CohortKey"]).startswith("sizesqft=")]
    assert len(company_grain) >= 45, "company-grain cohort rows unexpectedly few"
    for r in company_grain:
        sb = r["SizeBand"] if isinstance(r["SizeBand"], str) else None
        v = r["Vertical"] if isinstance(r["Vertical"], str) else None
        assert cv.cohort_key(sb, v) == r["CohortKey"], \
            f"parity break: ({sb},{v}) -> {cv.cohort_key(sb, v)} != {r['CohortKey']}"
        if sb:
            assert sb in cv.SIZE_BANDS, f"unknown live size band {sb}"
        if v:
            assert v in cv.VERTICALS, f"unknown live vertical {v}"
    # no NEW key families may appear silently — only size/vertical/global/sizesqft
    for r in df.to_dict("records"):
        k = str(r["CohortKey"])
        assert k == "global" or k.startswith(("size=", "vertical=", "sizesqft=")), \
            f"unrecognized cohort key family: {k}"


def test_spacedemand_score_universe_shared():
    """score_universe() is the single scoring pass: full-universe frame with calibrated
    propensity, and leads() must be a pure filter/rank view over it (no re-scoring drift)."""
    if not _db_ok():
        raise Skip("no DB")
    from contracts.signal_desk_spacedemand import data as sd
    df, b = sd.score_universe()
    assert len(df) > 1500, f"scored universe too small ({len(df)})"
    assert "propensity" in df.columns and df["propensity"].between(0, 1).all()
    out = sd.leads(market="IA,NE", limit=5)
    assert out["count"] == 5 and out["model_version"] == b.get("version")
    in_mkt = df[df["State"].isin({"Iowa", "Nebraska", "IA", "NE"})]
    assert out["leads"][0]["propensity"] == round(float(in_mkt["propensity"].max()), 4), \
        "top lead propensity != universe max (leads no longer a view over score_universe?)"


def test_spacedemand_bulk_scores_feed():
    """scores() is the nightly persistence feed (Signal Desk rea.CompanyModelScore):
    full universe with no market filter, DSM/OMA accepted as market aliases, rows the
    same shape leads() serves, trained_at surfaced for staleness display."""
    if not _db_ok():
        raise Skip("no DB")
    from contracts.signal_desk_spacedemand import data as sd
    df, b = sd.score_universe()
    out = sd.scores()
    assert out["count"] == len(df), "scores() must return the FULL universe by default"
    assert out["trained_at"], "trained_at missing (Signal Desk request #7)"
    assert out["scores"][0]["propensity"] >= out["scores"][-1]["propensity"]
    lead_keys = set(sd.leads(limit=1)["leads"][0])
    assert set(out["scores"][0]) == lead_keys, "scores row shape drifted from leads rows"
    dsm = sd.scores(market="DSM,OMA")
    ia = sd.scores(market="IA,NE")
    assert dsm["count"] == ia["count"] > 0, "DSM/OMA aliases must resolve to IA/NE"


def test_concession_offer_economics_offline():
    """Pure offer math (no DB): costs, breakevens, C_max, applicability, scenario mode."""
    from contracts.multifamily_churn import concession_cohort as cc
    p, tc, rent = 0.65, 5097.0, 1817.0
    e = cc.offer_economics(p, tc, rent, has_garage=False, garage_rent_monthly=0.0,
                           vacant_stalls=20, vacant_avg_rent=86.25, renewal_increase_pct=3.72)
    o = e["offers"]
    assert abs(o["free_months_1"]["cost_if_renewed"] - rent) < 1
    be = (1 - p) * rent / (tc - rent) * 100
    assert abs(o["free_months_1"]["breakeven_uplift_pp"] - be) < 0.15
    assert abs(o["rent_discount_5pct"]["cost_if_renewed"] - 0.6 * rent) < 1
    assert o["rent_discount_5pct"]["cost_if_baked_into_base_3yr"] > o["rent_discount_5pct"]["cost_if_renewed"] * 2.9
    assert o["free_garage_new"]["applicable"] and not o["free_garage_waiver"]["applicable"]
    assert abs(o["free_garage_new"]["cost_if_renewed"] - (12 - 2) * 86.25) < 1
    # cheaper offers never carry a higher breakeven
    live = sorted((v["cost_if_renewed"], v["breakeven_uplift_pp"]) for v in o.values() if v["applicable"])
    assert all(a[1] <= b[1] for a, b in zip(live, live[1:])), live
    assert e["cheapest_breakeven_offer"] == "free_garage_new"
    assert abs(e["c_max"]["at_10pp"] - 0.10 * tc / (1 - p + 0.10)) < 1
    assert e["recommended_offer_at_assumed_uplift"] is None
    # garage holder -> waiver applicable at 12 x their rent, new assignment not
    g = cc.offer_economics(p, tc, rent, has_garage=True, garage_rent_monthly=100.0,
                           vacant_stalls=20, vacant_avg_rent=86.25)
    assert g["offers"]["free_garage_waiver"]["applicable"] and not g["offers"]["free_garage_new"]["applicable"]
    assert abs(g["offers"]["free_garage_waiver"]["cost_if_renewed"] - 1200) < 1
    # never: cost >= turnover cost
    n = cc.offer_economics(p, 1500.0, rent, has_garage=False)
    assert n["offers"]["free_months_1"]["verdict"] == "never"
    assert n["offers"]["free_months_1"]["breakeven_uplift_pp"] is None
    # post-notice-like p collapses the sleeping-dog term
    pn = cc.offer_economics(0.973, tc, rent, has_garage=False)
    assert pn["offers"]["free_months_1"]["breakeven_uplift_pp"] < 2.0
    # scenario mode is labeled and arithmetically exact
    s = cc.offer_economics(p, tc, rent, has_garage=False, assumed_uplift=0.10)
    net = 0.10 * tc - (1 - p + 0.10) * rent
    assert abs(s["offers"]["free_months_1"]["expected_net_at_assumed_uplift"] - net) < 1
    assert s["recommended_offer_at_assumed_uplift"] in set(cc.OFFERS) | {"none"}
    # bad inputs rejected at the view boundary
    for kw in ({"band_low": 0.7, "band_high": 0.6}, {"assumed_uplift": 1.5},
               {"decision_window_days": -1}, {"garage_months_to_rerent": 13}):
        try:
            cc.build([82], **kw)
            raise AssertionError(f"accepted bad input {kw}")
        except ValueError:
            pass


def test_concession_cohort_view():
    """The intern surface: band + window + no-notice respected, identifier-only rows,
    full feature vector, ledger-coherent economics, CSV round-trip."""
    if not _db_ok():
        raise Skip("no DB")
    from contracts.multifamily_churn import concession_cohort as cc, data as mfd
    # the agreed persuadable band (0.20-0.60) is the non-empty anchor for data checks;
    # the default band (now also 0.20-0.60) is checked for SHAPE only on one property
    out = cc.build(None, band_low=0.20, band_high=0.60, decision_window_days=180)
    assert out["view"] == "concession-cohort" and out["model_version"].endswith("h180")
    d = cc.build([82])
    for k in ("n", "shown", "cohort", "offers_summary", "context", "members", "summary"):
        assert k in d, f"default-band response missing {k}"
    assert out["n"] == len(out["members"]) == out["shown"]
    assert out["population"]["scored_n"] > 500
    from core import registry as _reg
    nfeat = len(_reg.load("multifamily_churn", cc.RENEWAL_BUNDLE)["feature_columns"])
    for m in out["members"]:
        assert "full_name" not in m and "name" not in m, "rows must be identifier-only"
        assert 0.20 <= m["model"]["churn_probability"] < 0.60
        assert m["lease"]["has_notice"] == 0
        assert m["lease"]["days_to_lease_end"] is None or m["lease"]["days_to_lease_end"] <= 180
        assert abs(m["model"]["stay_probability"] + m["model"]["churn_probability"] - 1) < 1e-6
        assert len(m["features"]) == nfeat, (len(m["features"]), nfeat)
        e = m["economics"]
        if e["turnover_cost_net"]:
            assert abs(e["expected_loss"] - m["model"]["churn_probability"] * e["turnover_cost_net"]) < 1
            for k, o in e["offers"].items():
                if o["applicable"] and o["breakeven_uplift_pp"] is not None:
                    p, C, tc = m["model"]["churn_probability"], o["cost_if_renewed"], e["turnover_cost_net"]
                    assert abs(o["breakeven_uplift_pp"] - (1 - p) * C / (tc - C) * 100) < 0.6, (k, o)
                    assert "expected_net_at_assumed_uplift" not in o, "no uplift may be assumed by default"
        assert (m["garage"]["has_garage"] is True) == (m["garage"]["stalls_n"] > 0)
    # roll-ups cover the whole cohort even when top trims the display
    t2 = cc.build(None, band_low=0.20, band_high=0.60, top=3)
    assert t2["n"] == out["n"] and t2["shown"] == min(3, out["n"])
    assert t2["cohort"]["n"] == out["cohort"]["n"]
    # every property with garage inventory reports capacity; ledger context present
    assert set(out["context"]["ledger"]) == set(mfd.DEFAULT_PROPERTY_IDS)
    assert out["context"]["garage_capacity"][82]["stalls"] == 137
    # scenario mode adds labeled fields only when asked
    sc = cc.build([82], band_low=0.20, band_high=0.60, assumed_uplift=0.10, top=5)
    assert "SCENARIO" in sc["assumptions"]["uplift"]
    for m in sc["members"]:
        for o in m["economics"]["offers"].values():
            if o["applicable"]:
                assert "expected_net_at_assumed_uplift" in o
    # CSV: header + one line per shown member, flattened offer columns present
    csv_text = cc.to_csv(out)
    lines = csv_text.strip().splitlines()
    assert len(lines) == out["shown"] + 1
    assert "economics.offers.free_months_1.breakeven_uplift_pp" in lines[0]
    assert "features.TenureMonths" in lines[0]
    # a cohort must exist somewhere in the portfolio for the intern to work with
    assert out["n"] > 0, out["summary"]


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = skipped = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Skip as s:
            print(f"  SKIP  {t.__name__} ({s})")
            skipped += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {skipped} skipped, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run())
