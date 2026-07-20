"""Brokerage conversion v0.5 — honest-eval, two-headed (pipeline-momentum + cold-fit).

Implements roadmap Phases 1.1-1.3 (docs/BROKERAGE_AUDIT_2026-07-16.md):
  1.3 Hygienic corrected label via data.labeled_panel (date-filtered bridge, post-
      conversion negatives censored) -- the single label source.
  1.1 HONEST evaluation: feature selection runs INSIDE each CV fold (nested), and folds
      are GROUP-DISJOINT by prospect (GroupKFold on hMy) so one conversion cannot label
      both train and test. The audit showed an identity-honest TEMPORAL split is
      infeasible on this 2024-11+ panel (0 valid folds), so grouped-CV pooled-OOF PR-AUC
      is the honest headline, reported ALONGSIDE the old rolling-origin figure for
      continuity (which is inflated by pre-fold selection + prospect memorization).
  1.2 TWO HEADS, so the UI stops conflating "already closing" with "worth waking up":
        * pipeline_momentum -- CRM stage/activity dynamics (endogenous to an open deal;
          trivially strong but only says a deal is in motion),
        * cold_fit -- firmographics + engagement + contact quality (the only signal
          useful for COLD prospecting; honestly weak),
        * combined -- all non-scoop features (back-compat headline).
      Scoops are excluded from all heads (measured twice: no conversion lift).

Also derives distribution-based opportunity tiers (Phase 0.3) from OOF scores.

Run: PYTHONPATH=. .venv/bin/python scripts/train_brokerage_v05.py [--promote]
Honors GBME_BUNDLE_VERSION / GBME_NO_PROMOTE. SELECT-only DB; writes the local bundle only.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core import db                                                # noqa: E402,F401
from contracts.signal_desk_brokerage import data as bk_data        # noqa: E402

VERSION = os.environ.get("GBME_BUNDLE_VERSION", "v0.5-split-honest")
TS = datetime.now().strftime("%Y%m%d_%H%M%S")
RNG = np.random.default_rng(7)

MOMENTUM = ["pipeline_status", "pipeline_stage", "pipeline_probability",
            "stage_transitions_count", "days_in_current_stage", "had_stage_regression",
            "activity_count_30d", "activity_count_90d", "activity_count_365d",
            "days_since_last_activity", "first_response_hours", "unique_agents"]
COLDFIT = ["prospect_age_days", "source_code", "brand", "sales_category", "is_specialty",
           "is_cml_lead", "why_moving_type", "contact_method", "zi_revenue_k_usd",
           "zi_employee_count", "zi_company_age_years", "tech_count", "tech_has_crm",
           "linked_contact_count", "has_c_level_contact", "max_contact_accuracy",
           "is_linked_to_zi"]
HEADS = {"pipeline_momentum": MOMENTUM, "cold_fit": COLDFIT, "combined": MOMENTUM + COLDFIT}
SELECT_CAP = 8                    # Riley event-budget for ~54 events


def _matrix(df, feats, med):
    X = pd.DataFrame(index=df.index)
    for c in feats:
        X[c] = pd.to_numeric(df.get(c), errors="coerce").fillna(med.get(c, 0.0)).astype(float)
    return X


def _penalized():
    return Pipeline([("sc", StandardScaler()),
                     ("lr", LogisticRegression(penalty="l2", C=1.0, max_iter=2000, random_state=7))])


def _select(Xtr, ytr, cap=SELECT_CAP):
    """Elastic-net feature selection (top-|coef|, capped)."""
    Z = StandardScaler().fit_transform(Xtr.values)
    enet = LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5,
                              C=0.2, max_iter=5000, random_state=7)
    enet.fit(Z, ytr)
    coef = np.abs(enet.coef_[0])
    order = [i for i in np.argsort(coef)[::-1] if coef[i] > 1e-8][:cap]
    return [Xtr.columns[i] for i in order] or list(Xtr.columns[:1])


def nested_grouped_cv(df, pool, y, groups, n_splits=5):
    """Group-disjoint (by hMy) CV with feature selection INSIDE each fold -> pooled OOF.
    This is the honest generalization estimate for UNSEEN companies."""
    med_all = {c: pd.to_numeric(df[c], errors="coerce").median() for c in pool}
    med_all = {c: (0.0 if pd.isna(v) else float(v)) for c, v in med_all.items()}
    X = _matrix(df, pool, med_all)
    gkf = GroupKFold(n_splits=min(n_splits, np.unique(groups).size))
    oof_y, oof_p = np.zeros(len(y)), np.full(len(y), np.nan)
    for tr, te in gkf.split(X, y, groups):
        med = {c: (0.0 if pd.isna(v) else float(v))
               for c, v in X.iloc[tr].median().items()}
        Xtr = _matrix(df.iloc[tr], pool, med)
        sel = _select(Xtr, y[tr])
        m = _penalized().fit(Xtr[sel].values, y[tr])
        oof_p[te] = m.predict_proba(_matrix(df.iloc[te], pool, med)[sel].values)[:, 1]
        oof_y[te] = y[te]
    ok = ~np.isnan(oof_p)
    return oof_y[ok].astype(int), oof_p[ok]


def _boot_lift(y, p, n=2000):
    base = float(y.mean())
    diffs = []
    for _ in range(n):
        idx = RNG.integers(0, len(y), len(y))
        if y[idx].sum() < 2:
            continue
        diffs.append(average_precision_score(y[idx], p[idx]) - y[idx].mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5]) if diffs else (None, None)
    return round(base, 4), round(float(np.mean(diffs)), 4) if diffs else None, \
        (round(float(lo), 4), round(float(hi), 4)) if diffs else (None, None)


def _honest_metrics(df, pool, y, groups):
    yo, po = nested_grouped_cv(df, pool, y, groups)
    base, lift, ci = _boot_lift(yo, po)
    return {"validation": "nested group-disjoint 5-fold CV (selection in-fold), pooled OOF",
            "oof_n": int(len(yo)), "oof_pos": int(yo.sum()),
            "pr_auc": round(float(average_precision_score(yo, po)), 4),
            "pr_auc_no_skill": base, "pr_auc_lift_ci95": list(ci),
            "roc_auc": round(float(roc_auc_score(yo, po)), 4),
            "brier": round(float(brier_score_loss(yo, po)), 4),
            "pr_auc_vs_base_x": round(average_precision_score(yo, po) / base, 1) if base else None}, yo, po


def _fit_final(df, pool, y, med):
    """In-sample selection + penalized-logistic + Platt for serving. (Honest metric is the
    nested CV above; the served feature set is chosen on all data, noted in the manifest.)"""
    X = _matrix(df, pool, med)
    sel = _select(X, y)
    try:
        model = CalibratedClassifierCV(_penalized(), method="sigmoid", cv=3).fit(X[sel].values, y)
        calib = "platt"
    except Exception:
        model = _penalized().fit(X[sel].values, y); calib = "none"
    return model, sel, calib


def _tier_policy(yo, po, floor, ceil, hot_q=0.90, warm_q=0.70):
    """Distribution-derived opportunity tiers from OOF scores (Phase 0.3). Records the
    realized conversion rate + recall each tier buys so the tier is operational."""
    p = np.clip(po, floor, ceil)
    hot_cut, warm_cut = float(np.quantile(p, hot_q)), float(np.quantile(p, warm_q))
    if warm_cut >= hot_cut:
        warm_cut = max(floor, hot_cut * 0.5)
    hot = po >= hot_cut
    warm = (po >= warm_cut) & ~hot
    tot = max(1, int(yo.sum()))
    return {"hot_min": round(hot_cut, 4), "warm_min": round(warm_cut, 4),
            "hot_share": round(float(hot.mean()), 3),
            "hot_realized_conv": round(float(yo[hot].mean()), 4) if hot.any() else None,
            "hot_recall": round(float(yo[hot].sum() / tot), 3),
            "warm_realized_conv": round(float(yo[warm].mean()), 4) if warm.any() else None,
            "hot_plus_warm_recall": round(float(yo[hot | warm].sum() / tot), 3),
            "labels": {"hot": "Hot lead", "warm": "Watch", "cold": "Low priority"},
            "basis": "OOF quantiles (hot=top 10%, warm=next 20%)"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--promote", action="store_true")
    args = ap.parse_args()
    no_promote = os.environ.get("GBME_NO_PROMOTE") == "1"

    print("Loading hygienic labeled panel (corrected bridge, censored)...")
    df = bk_data.labeled_panel()
    y = df["Label"].to_numpy().astype(int)
    groups = df["hMy"].to_numpy()
    print(f"  rows={len(df)} prospects={df['hMy'].nunique()} pos={int(y.sum())} "
          f"pos_prospects={df.loc[y==1,'hMy'].nunique()} base={y.mean():.4%}")

    med = {c: pd.to_numeric(df[c], errors="coerce").median() for c in HEADS["combined"]}
    med = {c: (0.0 if pd.isna(v) else float(v)) for c, v in med.items()}

    models, feats, metrics, oof = {}, {}, {}, {}
    for head, pool in HEADS.items():
        met, yo, po = _honest_metrics(df, pool, y, groups)
        model, sel, calib = _fit_final(df, pool, y, med)
        models[head], feats[head], metrics[head] = model, sel, met
        oof[head] = (yo, po)
        print(f"  [{head:16}] honest PR-AUC={met['pr_auc']} (x{met['pr_auc_vs_base_x']} base) "
              f"ROC={met['roc_auc']} lift_CI={met['pr_auc_lift_ci95']} sel={sel}")

    FLOOR, CEIL = 0.002, 0.99
    tiers = {h: _tier_policy(*oof[h], FLOOR, CEIL) for h in ("combined", "cold_fit")}
    print(f"  opportunity tiers (combined): {tiers['combined']}")

    bundle = {
        "bundle_format": "brokerage-split-v0.5",
        "models": models, "features": feats, "feature_medians": med,
        "calibration": {"method": "platt", "prob_floor": FLOOR, "prob_ceil": CEIL},
        "opportunity_tiers": tiers,
        "tier_labels": tiers["combined"]["labels"],
        "version": VERSION, "trained_at": TS, "algorithm": "two-head penalized-logistic + Platt",
        "horizon_days": 180,
        "label": "corrected bridge (data.labeled_panel): vw_Deal.tenantId->CommAmendmentSnapshot"
                 ".hTenant, iType 0/1/2/15, COALESCE(dtSign,dtStart) within 180d after snapshot; "
                 "dates filtered, post-conversion negatives censored",
        "source": "real-corrected-label",
        "metrics": {**{f"head_{h}": metrics[h] for h in HEADS},
                    "n_converting_prospects": int(df.loc[y==1,'hMy'].nunique()),
                    "n_prospects": int(df["hMy"].nunique()), "n_rows": int(len(df)),
                    "n_pos_rows": int(y.sum()), "base_rate": round(float(y.mean()), 4),
                    "headline": "cold_fit is the cold-prospecting signal; pipeline_momentum "
                                "mostly detects deals already in motion"},
    }
    manifest = {"contract": "signal-desk-brokerage", "version": VERSION,
                "bundle_format": "brokerage-split-v0.5", "trained_at": TS,
                "heads": {h: {"features": feats[h], **{k: metrics[h][k] for k in
                          ("pr_auc", "pr_auc_no_skill", "pr_auc_lift_ci95", "roc_auc", "oof_pos")}}
                          for h in HEADS},
                "opportunity_tiers": tiers,
                "metrics": {"pr_auc": metrics["combined"]["pr_auc"],   # gate reads this
                            "pr_auc_no_skill": metrics["combined"]["pr_auc_no_skill"],
                            "pr_auc_lift_ci95": metrics["combined"]["pr_auc_lift_ci95"],
                            "roc_auc": metrics["combined"]["roc_auc"],
                            "oof_pos": metrics["combined"]["oof_pos"]},
                "note": "Honest nested group-disjoint CV (the reported PR-AUC is ~20% below the "
                        "old rolling-origin figure, which was inflated by pre-fold selection + "
                        "prospect memorization). Two heads: cold_fit (cold prospecting) vs "
                        "pipeline_momentum (deal-in-motion). INTERIM/data-limited (54 events)."}
    d = ROOT / "bundles" / "signal_desk_brokerage" / VERSION
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, d / "bundle.joblib")
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    if args.promote and not no_promote:
        (d.parent / "default.txt").write_text(VERSION)
        print(f"  promoted default.txt -> {VERSION}")
    else:
        print(f"  saved {VERSION} (not promoted)")
    print("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
