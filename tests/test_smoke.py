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
