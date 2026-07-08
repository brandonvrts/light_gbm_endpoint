"""Model-validation cards for the technical sector: training setup, discrimination
metrics (ROC-AUC variants), Brier/log-loss, calibration internals, feature importance.
All values are read from the SERVED artifact (bundle + manifest) so the card describes
exactly what is deployed. PR-AUC / log loss are not stored in the joblib; they come from
the leakage-audited v2 validation paper and are labeled as such.
"""
from __future__ import annotations
from core import registry

# Documented validation-paper figures (not in the joblib; brier/AUC ARE in the artifact).
_MF_PAPER = {"pr_auc": 0.945, "pr_auc_no_skill": 0.745, "log_loss": 0.291,
             "source": "v2 walk-forward validation paper (13 quarterly folds, leakage-audited)"}


def _mf_card() -> dict:
    b = registry.load("multifamily_churn")
    m = registry.manifest("multifamily_churn")
    cal = b.get("calibration") or {}
    met = b.get("metrics") or {}
    tw = b.get("train_window") or {}
    wf = met.get("walkforward") or {}
    return {
        "contract": "multifamily-churn", "status": "real-production", "label": "Real - validated",
        "version": b.get("version"), "algorithm": b.get("algorithm"),
        "horizon_days": b.get("horizon_days"), "trained_at": b.get("trained_at"),
        "data_source": b.get("data_source"), "feature_count": len(b.get("feature_columns", [])),
        "training": {
            "design": "Anchor-date: one (tenant, month-end anchor) row; label = move-out within horizon; "
                      "group key hMyPerson so a tenant never spans train/test.",
            "validation": "Quarterly walk-forward (train past, test next quarter) + GroupKFold; "
                          "group-disjoint, leakage-audited.",
            "n_samples": tw.get("n_samples"), "n_tenants": tw.get("n_tenants"),
            "churn_rate": tw.get("churn_rate"),
            "decision_date_min": tw.get("decision_date_min"), "decision_date_max": tw.get("decision_date_max"),
        },
        "discrimination": {
            "walkforward_auc_mean": wf.get("mean_auc"), "walkforward_auc_std": wf.get("std_auc"),
            "walkforward_folds": wf.get("n_folds"),
            "group_split_train_auc": met.get("group_split_train_auc"),
            "group_split_test_auc": met.get("group_split_test_auc"),
            "group_kfold_auc_mean": met.get("group_kfold_auc_mean"),
            "group_kfold_auc_std": met.get("group_kfold_auc_std"),
            "overfit_gap": met.get("overfit_gap"),
            "pr_auc": _MF_PAPER["pr_auc"], "pr_auc_no_skill": _MF_PAPER["pr_auc_no_skill"],
            "log_loss": _MF_PAPER["log_loss"], "paper_source": _MF_PAPER["source"],
        },
        "brier": {"raw": cal.get("brier_raw"), "sigmoid": cal.get("brier_sigmoid"),
                  "isotonic": cal.get("brier_isotonic"), "post_recal": cal.get("brier_post_recal")},
        "calibration": {
            "method": "isotonic (shape) + Platt anchor (level)",
            "base_rate_target": cal.get("base_rate_target"),
            "mean_pred_pre": cal.get("mean_pred_pre"), "mean_pred_post": cal.get("mean_pred_post"),
            "platt_a": cal.get("post_hoc_platt_a"), "platt_b": cal.get("post_hoc_platt_b"),
            "prob_floor": cal.get("prob_floor"), "prob_ceil": cal.get("prob_ceil"),
            "calibration_set_size": cal.get("calibration_set_size"),
            "optimal_threshold": cal.get("optimal_threshold"), "optimal_f1": cal.get("optimal_f1"),
            "optimal_precision": cal.get("optimal_precision"), "optimal_recall": cal.get("optimal_recall"),
        },
        "feature_importance": (b.get("feature_importance") or [])[:15],
        "notes": m.get("notes") or m.get("note"),
    }


def _bk_card() -> dict:
    b = registry.load("signal_desk_brokerage")
    m = registry.manifest("signal_desk_brokerage")
    cal = b.get("calibration") or {}
    met = b.get("metrics") or {}
    if b.get("bundle_format") == "standin":   # legacy leaky stand-in (kept for rollback)
        return {
            "contract": "signal-desk-brokerage", "status": "interim-standin",
            "label": "Interim stand-in - NOT walk-forward validated (label is broken)",
            "version": b.get("version"), "algorithm": b.get("algorithm"),
            "feature_count": len(b.get("feature_columns", [])),
            "discrimination": {"holdout_auc": m.get("train_auc"),
                               "caveat": "The 0.993 AUC is meaningless leakage: ProspectLabelSnapshot is "
                                         "keyed on iType=13 proposals, marking all snapshots of every converter."},
            "notes": "Leaky stand-in. Superseded by the corrected-label model; kept only for rollback.",
        }
    # Corrected, time-localized label (deal->amendment bridge); small-n-defensible logistic.
    w = met.get("winner") or {}
    pen, fir = met.get("penalized") or {}, met.get("firth") or {}
    ci = w.get("pr_auc_lift_ci95") or []
    sel = met.get("selected_features") or b.get("feature_columns") or []
    return {
        "contract": "signal-desk-brokerage", "status": "interim-corrected",
        "label": "Corrected bridge label - rolling-origin validated, DATA-LIMITED (interim)",
        "version": b.get("version"), "algorithm": b.get("algorithm"),
        "feature_count": len(b.get("feature_columns", [])),
        "training": {
            "design": "Time-localized label via the deal->amendment bridge (rea.vw_Deal.tenantId -> "
                      "dbo.CommAmendmentSnapshot.hTenant; iType 0/1/2/15; COALESCE(dtSign,dtStart) within 180d "
                      f"after each snapshot). Scoop-free; elastic-net selected {len(sel)} of "
                      f"{met.get('feature_pool')} features; penalized-logistic vs Firth+FLIC, no resampling.",
            "selected_features": sel,
            "validation": w.get("validation"),
            "n_rows": met.get("n_rows"), "n_pos_rows": met.get("n_pos_rows"),
            "n_prospects": met.get("n_prospects"),
            "n_converting_prospects": met.get("n_converting_prospects"),
            "base_rate": met.get("base_rate"),
        },
        "discrimination": {
            "pr_auc": w.get("pr_auc"), "pr_auc_no_skill": w.get("pr_auc_no_skill"),
            "pr_auc_lift_ci95": ci, "roc_auc": w.get("roc_auc"),
            "caveat": "PR-AUC with a paired-bootstrap CI vs no-skill is the honest read at this ~0.8% base "
                      "rate; ROC-AUC is imbalance-inflated. The CI excludes 0 (real signal) but is wide - "
                      "few converting prospects.",
        },
        "model_selection": {"chosen": w.get("estimator"),
                            "penalized_pr_auc": pen.get("pr_auc"), "firth_pr_auc": fir.get("pr_auc")},
        "brier": {"oof": w.get("brier")}, "log_loss": w.get("log_loss"),
        "calibration": {"method": cal.get("method"), "prob_floor": cal.get("prob_floor"),
                        "prob_ceil": cal.get("prob_ceil"),
                        "intercept": w.get("calib_intercept"), "slope": w.get("calib_slope")},
        "walk_forward": {
            "possible": "interim",
            "status": f"Expanding-window rolling-origin ({w.get('test_folds')} quarterly test folds), pooled "
                      f"OOF PR-AUC {w.get('pr_auc')} (no-skill {w.get('pr_auc_no_skill')}), lift CI95 {ci}. "
                      f"Real but data-limited: {met.get('n_converting_prospects')} converting prospects.",
            "next": "UPDATE 2026-06-25: zi.Scoop was backfilled (109k rows, 2022-2026), so re-add the scoop "
                    "family to this prospect model (v0.4). Binding limiter is now bridge coverage (~46% via "
                    "rea.ProspectCompanyLink) + label scarcity. Sample size via Riley/pmsampsize, not 10-EPV.",
            "fix_spec": "light_gbm_endpoint/docs/SIGNAL_DESK_DATA_VALIDATION.md",
        },
        "notes": "INTERIM, data-limited, on a CORRECT (non-leaky) label - replaces the 0.993-AUC leaky "
                 "stand-in and the v0.2 LightGBM. Estimator chosen between penalized-logistic and Firth+FLIC "
                 "by rolling-origin OOF PR-AUC. Directional signal alongside the rules score; not for "
                 "autonomous decisions until the converting-prospect count grows.",
    }


def _mk_card() -> dict:
    return {
        "contract": "marketing", "status": "aggregate",
        "label": "Cohort-opportunity model - grounded ranking (no per-entity classifier)",
        "training": {"design": "Ranks cohorts by expected_winnable = addressable prospects (mkt.CohortCatalog "
                               "ProspectsTagged) x historical win-rate (real won/deals), enriched with a recent "
                               "scoop-momentum signal per industry (zi.Scoop 2022-2026 via rea.ProspectCompanyLink)."},
        "discrimination": {"caveat": "Aggregate ranking, not a supervised per-company classifier - there is no "
                                     "marketing-scale conversion label. Win-rate + addressable + scoop momentum "
                                     "are all directly observed, not predicted."},
        "notes": "Served real-time at GET /marketing/targeting (cohorts + verticals + scoop_momentum_90d). "
                 "$ EV intentionally omitted - rea.DealPattern median rent is unpopulated, so a dollar value "
                 "would be fabricated. Add it once deal economics are backfilled.",
    }


def _dealwin_card() -> dict | None:
    try:
        b = registry.load("signal_desk_dealwin")
    except Exception:
        return None
    met = b.get("metrics") or {}
    return {
        "contract": "signal-desk-dealwin", "status": "shadow",
        "label": "Deal-win re-ranker (SHADOW) - ranking-grade, not broker-facing",
        "version": b.get("version"), "algorithm": b.get("algorithm"), "grain": "deal",
        "feature_count": len(b.get("feature_columns", [])),
        "training": {
            "design": "Resolved Yardi deals (StageResolved=Executed -> won vs Cancelled/Dead Deal -> lost); "
                      "features cohortRate (k=12 shrink, TRAIN-ONLY) + sqft band + deal type + leadBroker; "
                      "point-in-time scoop features (2022-2026 backfill) tested as an add-on.",
            "resolved_deals": met.get("resolved_deals"), "won": met.get("won"),
            "win_rate": met.get("win_rate"), "with_zicompany": met.get("with_zicompany"),
        },
        "discrimination": {
            "base_auc": met.get("base_mean_auc"), "baseline_auc": met.get("baseline_mean_auc"),
            "base_auc_lift": met.get("base_mean_diff"),
            "scoops_auc": met.get("scoops_mean_auc"), "scoops_auc_lift": met.get("scoops_mean_diff"),
            "caveat": "Reproduces the app's shadow edge (~+0.07 AUC over the cohort baseline; lift CI excludes "
                      "0 on the 2 larger resolved-date splits). Scoop features do NOT help at the deal grain "
                      "(only ~122/906 deals have a pre-open scoop) - that richness is for the prospect/marketing models.",
        },
        "validation": "corrected resolved-date split (ModifiedAt) + train-only cohort prior + paired-bootstrap AUC-lift CI + ECE",
        "walk_forward": {
            "possible": "shadow-failed-gate",
            "status": "GRADUATION FAILED (2026-06-25). Ablating leadBroker collapses the model to the cohort "
                      "baseline exactly (ablated-vs-cohort AUC diff 0.0, CI [0,0]) - the entire +0.069 edge IS "
                      "leadBroker. And leadBroker prevalence shifts ~7% (train) -> ~40% (test): a broker-"
                      "selection era-proxy, not a company fundamental. ECE ~0.09-0.14. STAYS SHADOW; not served.",
            "next": "Needs features with edge BEYOND cohort+leadBroker. Scoops don't help at deal grain; "
                    "point-in-time concept tags (B2 snapshot track) are the candidate. Until then the served "
                    "brokerage model is the cold-prospect conversion model, not this deal re-ranker.",
        },
        "notes": "Ported from the Signal Desk shadow win-model; reproduces its in-sample +0.069 AUC edge but "
                 "that edge is entirely leadBroker (broker-conviction proxy, era-shifted). Research/shadow only - "
                 "measured, NOT shown to brokers and NOT a served prediction.",
    }


def build_all() -> list:
    out = []
    for fn in (_mf_card, _bk_card, _mk_card, _dealwin_card):
        try:
            c = fn()
            if c:
                out.append(c)
        except Exception as e:  # noqa: BLE001
            out.append({"error": str(e)[:200]})
    return out
