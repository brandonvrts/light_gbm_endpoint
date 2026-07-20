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
