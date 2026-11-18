"""
Train + honestly evaluate the renewal-decision churn model on the corrected
point-in-time dataset (pit_training.py output).

Evaluation protocol (the "realistic AUC")
-----------------------------------------
Walk-forward by calendar quarter of the ANCHOR date:
  * test block  = all samples with anchor in quarter Q, deduplicated to ONE
    row per (resident, lease-end) at the anchor closest to 90 days before
    lease end (mirrors the monthly report's offer-time scoring);
  * train block = samples whose outcome window (lease_end + 90d) closed
    BEFORE the test block starts (no label peeking), weighted 1/k per
    lease-end event (k = number of anchors the event appears at).
Pooled out-of-time predictions across all folds give the headline AUC with a
bootstrap CI. The existing v4 bundle is scored on the IDENTICAL test rows as
the baseline.

Models compared (default run)
-----------------------------
  v4_baseline   : production churn_model_v4_h180 raw scores (no retrain)
  v4_leaky      : retrain on v4 feature set INCLUDING DTNOTICE-derived
                  HasNotice/DaysSinceNotice (reference for the notice swap)
  v4_noleak     : v4 minus DTNOTICE features + PIT-safe tenant_history notice
                  (X_ev_notice_active/days_since)   [ask #3]
  v4_noleak+new : + subsidy (X_sub_*), concession burn-off (X_conc_*), and
                  unit attributes baths/floorplan/amenities (X_uatt_*)  [ask #2]

The headline metric is the "undecAUC" column: AUC on the PRE-NOTICE subset
(X_ev_notice_active==0), i.e. residents who have NOT yet given notice at the
anchor -- the population the forecast must actually rank.  Full-population AUC
is inflated by the notice signal and is reported only for continuity.

Usage:
    python train_renewal_model.py            # default 3-way comparison
    python train_renewal_model.py --ablate   # add-one per-family lift
    python extra_features.py                  # (run first to build the extra CSV)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import LabelEncoder

from ._cfg import DATA_DIR, MODEL_DIR

RNG = np.random.RandomState(42)

# The two DTNOTICE-derived columns (HasNotice, DaysSinceNotice) come from the
# single mutable TENANT.DTNOTICE field: at a HISTORICAL anchor it reflects the
# resident's *latest* notice, not their notice status as-of that anchor, so it
# leaks/misreports for training rows.  Dropped from the modeled set (2026-07-09);
# the PIT-correct replacement is the event-sourced tenant_history notice
# (X_ev_notice_active / X_ev_days_since_notice) from extra_features.py.
# Current (mutable) TENANT columns: iNSFCount / BNOPAYMENTS reflect TODAY, not
# the anchor, so at a historical anchor they can absorb post-anchor events.
# Excluded from the modelled set; X_pay_nsf_cnt (TRANS, evaluated at the anchor)
# is the PIT-safe stand-in.  2026-11-10.
NOT_PIT_SAFE = ['NSFCount', 'NoPaymentsFlag']
PIT_SAFE_SUBS = ['X_pay_nsf_cnt']

LEAKY_NOTICE = ['HasNotice', 'DaysSinceNotice']
NOTICE_SAFE = ['X_ev_notice_active', 'X_ev_days_since_notice']

V4_FEATURES = [
    'TenureDaysAtAnchor', 'TenureMonths', 'DaysToLeaseEnd', 'PlannedLeaseDays',
    'IsLeaseExpiringSoon', 'LeaseTermMonths', 'DecisionNumber',
    'CurrentRent', 'ActiveRent', 'PriorRent', 'RentIncreasePct',
    'NSFCount', 'NoPaymentsFlag',
    'MoveInMonth', 'MoveInYear', 'DecisionMonth', 'DecisionYear',
    'WorkOrderCount', 'WorkOrderFrequency', 'EmergencyWorkOrders',
    'AvgDaysToComplete', 'WorkOrders_Last90Days', 'WorkOrders_Last180Days',
    'WorkOrders_Last365Days', 'EmergencyWO_Last90Days', 'DaysSinceLastWorkOrder',
    'OpenWorkOrdersAtAnchor', 'MaxOpenWorkOrderAge', 'EarlyWorkOrders30d',
    'MemoCount', 'MemoFrequency', 'Memos_Last90Days', 'Memos_Last180Days',
    'Memos_Last365Days', 'DaysSinceLastMemo', 'EvictionMemos', 'PaymentMemos',
    'ComplaintMemos', 'MaintenanceMemos', 'PetCount', 'HasLargePets',
    'PropertyName_encoded', 'PrimaryPetType_encoded',
]

NEW_FEATURES = [
    'PriorLeaseCount', 'FirstRent', 'RentVsFirstPct', 'DaysSincePriorLease',
]

# res-chat-derived trial families (2026-07-09): subsidy, concession burn-off,
# unit attributes (baths/floorplan/amenities).  Evaluated for lift here before
# any promotion to the deployed pit_features/bundle path.
TRIAL_PREFIXES = ('X_sub_', 'X_conc_', 'X_uatt_')
# any additional engineered columns (delinquency, market, household...) are
# picked up automatically if present in the CSV with the EXTRA_ prefix
EXTRA_PREFIX = 'X_'

LGBM_PARAMS = dict(
    n_estimators=400, learning_rate=0.03, num_leaves=15, max_depth=5,
    min_child_samples=40, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=5.0,
    random_state=42, verbosity=-1,
)


def load_best_params():
    p = os.path.join(MODEL_DIR, 'best_params.json')
    if os.path.exists(p):
        try:
            j = json.load(open(p))
            params = j.get('params', j)
            keep = {k: v for k, v in params.items()
                    if k in ('n_estimators', 'learning_rate', 'num_leaves', 'max_depth',
                             'min_child_samples', 'subsample', 'colsample_bytree',
                             'reg_alpha', 'reg_lambda')}
            if keep:
                out = dict(LGBM_PARAMS); out.update(keep)
                return out
        except Exception:
            pass
    return LGBM_PARAMS


def prepare(df: pd.DataFrame):
    df = df.copy()
    # Ambiguous transfer cases: move-out inside the churn window AND evidence
    # of a follow-on lease (unit transfer / renew-then-leave). ~1% of rows;
    # label is undecidable -- exclude (per 2026-07-08 data-quality audit).
    amb = (df['ChurnedAtLeaseEnd'] == 1) & (df.get('RenewEvidence', 0) == 1)
    if amb.any():
        print(f"[clean] excluded {int(amb.sum())} ambiguous transfer rows "
              f"(churn-window move-out + follow-on lease)")
        df = df[~amb]
    df['PropertyName'] = df['PropertyName'].fillna('Unknown').astype(str)
    df['PrimaryPetType'] = df['PrimaryPetType'].fillna('None').astype(str)
    df['PropertyName_encoded'] = LabelEncoder().fit_transform(df['PropertyName'])
    df['PrimaryPetType_encoded'] = LabelEncoder().fit_transform(df['PrimaryPetType'])
    df['DaysToLeaseEnd'] = pd.to_numeric(df['DaysToLeaseEnd'], errors='coerce').clip(-365, 1825)
    df['PlannedLeaseDays'] = pd.to_numeric(df['PlannedLeaseDays'], errors='coerce').clip(0, 3650)
    for c in df.columns:
        if c in ('AnchorDate',):
            continue
        if df[c].dtype == object:
            continue
    df['AnchorDate'] = pd.to_datetime(df['AnchorDate'], format='mixed')
    df['AnchorLeaseToDate'] = pd.to_datetime(df['AnchorLeaseToDate'], format='mixed')
    df['event_id'] = df['hMyPerson'].astype(str) + '|' + df['AnchorLeaseToDate'].dt.strftime('%Y-%m-%d')
    k = df.groupby('event_id')['event_id'].transform('count')
    df['sample_weight'] = 1.0 / k
    return df


def feature_matrix(df, cols):
    X = df.reindex(columns=cols).apply(pd.to_numeric, errors='coerce')
    med = X.median(numeric_only=True)
    return X.fillna(med), med


def quarters(dates: pd.Series):
    q = dates.dt.to_period('Q')
    return sorted(q.unique())


def dedup_test(block: pd.DataFrame) -> pd.DataFrame:
    """One row per lease-end event: anchor closest to 90d before lease end."""
    block = block.copy()
    block['lead_gap'] = (block['DaysToLeaseEnd'] - 90).abs()
    return block.sort_values('lead_gap').groupby('event_id', as_index=False).first()


def bootstrap_ci(y, p, n=2000, seed=42):
    rng = np.random.RandomState(seed)
    y = np.asarray(y); p = np.asarray(p)
    idx = np.arange(len(y))
    aucs = []
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[s])) < 2:
            continue
        aucs.append(roc_auc_score(y[s], p[s]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def load_v4_bundle():
    path = sorted(glob.glob(os.path.join(MODEL_DIR, 'churn_model_v4_h180_recal_*.joblib')))[-1]
    b = joblib.load(path)

    def score(df_test):
        # v4 expects its own encoders + scaler + medians
        d = df_test.copy()
        prop_enc, pet_enc = b['property_encoder'], b['pet_encoder']
        known_p = set(prop_enc.classes_); fb_p = prop_enc.classes_[0]
        d['PropertyName'] = d['PropertyName'].fillna(fb_p).astype(str).apply(
            lambda x: x if x in known_p else fb_p)
        d['PropertyName_encoded'] = prop_enc.transform(d['PropertyName'])
        known_q = set(pet_enc.classes_); fb_q = 'None' if 'None' in known_q else pet_enc.classes_[0]
        d['PrimaryPetType'] = d['PrimaryPetType'].fillna(fb_q).astype(str).apply(
            lambda x: x if x in known_q else fb_q)
        d['PrimaryPetType_encoded'] = pet_enc.transform(d['PrimaryPetType'])
        X = d.reindex(columns=b['feature_columns']).apply(pd.to_numeric, errors='coerce')
        for c, m in (b.get('feature_medians') or {}).items():
            if c in X:
                X[c] = X[c].fillna(m)
        X = X.fillna(0)
        Xs = b['scaler'].transform(X)
        cc = getattr(b['model'], 'inner_model', b['model'])
        return cc.calibrated_classifiers_[0].estimator.predict_proba(Xs)[:, 1]
    return score, os.path.basename(path)


def walk_forward(df, feature_sets, params, include_v4=True, verbose=True):
    """Returns (summary, fold_rows, pool) where pool is a DataFrame of pooled
    out-of-time test rows with one p_<model> column per model."""
    v4_score = None
    if include_v4:
        v4_score, _ = load_v4_bundle()
    qs = quarters(df['AnchorDate'])
    model_names = list(feature_sets) + (['v4_baseline'] if include_v4 else [])
    fold_rows, pool_frames = [], []
    fold_aucs = {m: [] for m in model_names}

    for q in qs:
        test = df[df['AnchorDate'].dt.to_period('Q') == q]
        test = dedup_test(test)
        t_start = test['AnchorDate'].min()
        train = df[(df['AnchorLeaseToDate'] + pd.Timedelta(days=90)) < t_start]
        if len(train) < 400 or test['ChurnedAtLeaseEnd'].nunique() < 2 \
           or test['ChurnedAtLeaseEnd'].sum() < 5:
            continue
        y_te = test['ChurnedAtLeaseEnd'].values
        row = {'quarter': str(q), 'n_train': len(train), 'n_test': len(test),
               'churn_rate': float(y_te.mean())}
        keep_cols = ['event_id', 'AnchorDate', 'DaysToLeaseEnd', 'ChurnedAtLeaseEnd',
                     'PropertyName']
        for c in ('X_ev_notice_active', 'X_prpsl_selected', 'X_prpsl_exists'):
            if c in test.columns:
                keep_cols.append(c)
        tf = test[keep_cols].copy()
        tf['quarter'] = str(q)

        for name, cols in feature_sets.items():
            X_tr, med = feature_matrix(train, cols)
            X_te = test.reindex(columns=cols).apply(pd.to_numeric, errors='coerce').fillna(med)
            m = LGBMClassifier(**params)
            m.fit(X_tr, train['ChurnedAtLeaseEnd'], sample_weight=train['sample_weight'])
            p = m.predict_proba(X_te)[:, 1]
            tf[f'p_{name}'] = p
            auc = roc_auc_score(y_te, p)
            fold_aucs[name].append(auc)
            row[name] = round(auc, 4)

        if include_v4:
            pv4 = v4_score(test)
            tf['p_v4_baseline'] = pv4
            aucv4 = roc_auc_score(y_te, pv4)
            fold_aucs['v4_baseline'].append(aucv4)
            row['v4_baseline'] = round(aucv4, 4)
        fold_rows.append(row)
        pool_frames.append(tf)
        if verbose:
            print(f"  {q}  n_tr={len(train):>5d} n_te={len(test):>3d} churn={y_te.mean():.0%}  "
                  + "  ".join(f"{k}={row[k]:.3f}" for k in model_names))

    pool = pd.concat(pool_frames, ignore_index=True)
    y = pool['ChurnedAtLeaseEnd'].values
    undecided = np.ones(len(pool), dtype=bool)
    if 'X_ev_notice_active' in pool:
        undecided &= (pool['X_ev_notice_active'].fillna(0) == 0).values
    if 'X_prpsl_selected' in pool:
        undecided &= (pool['X_prpsl_selected'].fillna(0) == 0).values

    print(f"\n{'model':16s} {'pooledAUC':>9s} {'95% CI':>16s} {'foldAUC':>11s} "
          f"{'PR-AUC':>7s} {'Brier':>7s} {'undecAUC':>9s}")
    summary = {}
    for name in model_names:
        p = pool[f'p_{name}'].values
        auc = roc_auc_score(y, p)
        lo, hi = bootstrap_ci(y, p)
        pr = average_precision_score(y, p)
        br = brier_score_loss(y, p)
        u_auc = float('nan')
        if undecided.sum() > 50 and len(np.unique(y[undecided])) == 2:
            u_auc = roc_auc_score(y[undecided], p[undecided])
        summary[name] = {'pooled_auc': float(auc), 'ci': [lo, hi],
                         'fold_mean': float(np.mean(fold_aucs[name])),
                         'fold_std': float(np.std(fold_aucs[name])),
                         'pr_auc': float(pr), 'brier': float(br),
                         'n': int(len(y)), 'base_rate': float(y.mean()),
                         'undecided_auc': float(u_auc) if u_auc == u_auc else None,
                         'n_undecided': int(undecided.sum())}
        print(f"{name:16s} {auc:>9.4f} [{lo:.3f}, {hi:.3f}] "
              f"{np.mean(fold_aucs[name]):>6.3f}±{np.std(fold_aucs[name]):.3f} "
              f"{pr:>7.3f} {br:>7.4f} {u_auc:>9.3f}")
    print(f"(undecided subset: n={int(undecided.sum())}, "
          f"churn={y[undecided].mean():.1%})" if undecided.sum() else "")
    return summary, fold_rows, pool


FAMILY_PREFIXES = ['X_pay_', 'X_mkt_', 'X_prop_', 'X_unit_', 'X_ev_', 'X_prpsl_',
                   'X_hh_', 'X_sub_', 'X_conc_', 'X_uatt_']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=None)
    ap.add_argument('--extra', default=None,
                    help='pit_extra_features CSV to merge (default: newest, if any)')
    ap.add_argument('--ablate', action='store_true',
                    help='per-family add-one ablation instead of the 2-set run')
    args = ap.parse_args()
    csv = args.csv or sorted(glob.glob(os.path.join(DATA_DIR, 'pit_renewal_training_*.csv')))[-1]
    print(f"[load] {csv}")
    df = pd.read_csv(csv)

    extra = args.extra
    if extra is None:
        cands = sorted(glob.glob(os.path.join(DATA_DIR, 'pit_extra_features_*.csv')))
        extra = cands[-1] if cands else None
    if extra:
        xf = pd.read_csv(extra)
        xf['hMyPerson'] = pd.to_numeric(xf['hMyPerson']).astype('int64')
        df['hMyPerson'] = pd.to_numeric(df['hMyPerson']).astype('int64')
        df = df.merge(xf, on=['hMyPerson', 'AnchorDate'], how='left')
        print(f"[extra] merged {extra} "
              f"({sum(c.startswith('X_') for c in xf.columns)} features)")

    df = prepare(df)
    print(f"[data] samples={len(df)} events={df['event_id'].nunique()} "
          f"residents={df['hMyPerson'].nunique()} churn={df['ChurnedAtLeaseEnd'].mean():.1%}")

    v4nl = [c for c in V4_FEATURES if c in df.columns]         # v4 minus leaky notice
    v4_leaky = v4nl + [c for c in LEAKY_NOTICE if c in df.columns]  # reference (DTNOTICE)
    notice_safe = [c for c in NOTICE_SAFE if c in df.columns]  # PIT-safe replacement
    base_nl = v4nl + notice_safe                              # no-leak base (ask #3)
    trial_new = [c for c in df.columns if c.startswith(TRIAL_PREFIXES)]  # ask #2
    lh_feats = [c for c in NEW_FEATURES if c in df.columns]
    fam_cols = {p.rstrip('_').replace('X_', ''): [c for c in df.columns if c.startswith(p)]
                for p in FAMILY_PREFIXES}
    fam_cols = {k: v for k, v in fam_cols.items() if v}
    all_new = lh_feats + [c for cols in fam_cols.values() for c in cols]

    if args.ablate:
        # add-one ablation on the no-leak base; isolates each new family's lift
        feature_sets = {'base_nl': base_nl}
        for fam, cols in fam_cols.items():
            feature_sets[f'+{fam}'] = base_nl + cols
        feature_sets['+trial_new'] = base_nl + trial_new
        feature_sets['+all'] = base_nl + all_new
    else:
        feature_sets = {
            'v4_leaky':      v4_leaky,               # ref: current prod (DTNOTICE-derived notice)
            'v4_noleak':     base_nl,                # ask #3: PIT-safe tenant_history notice
            'v4_noleak+new': base_nl + trial_new,    # ask #2 + #3: + subsidy/concession/unit-attrs
        }
    # notice_safe lives in both base_nl and the 'ev' family, so ablate's +ev / +all
    # sets would duplicate those columns -- dedupe every set (order-preserving).
    feature_sets = {k: list(dict.fromkeys(v)) for k, v in feature_sets.items()}
    if notice_safe:
        print(f"[notice] dropped leaky {LEAKY_NOTICE} -> PIT-safe {notice_safe}")
    else:
        print(f"[notice][WARN] {NOTICE_SAFE} absent from extra CSV; run extra_features.py first")
    print(f"[features] v4_noleak={len(v4nl)}  notice_safe={len(notice_safe)}  "
          f"trial_new={len(trial_new)} ({trial_new})")

    params = load_best_params()
    print(f"[params] {params}")
    print('\nWalk-forward (quarterly, label-buffered, deduped test):')
    summary, folds, pool = walk_forward(df, feature_sets, params)

    ts = pd.Timestamp.now().strftime('%Y%m%d_%H%M')
    out = os.path.join(MODEL_DIR, f'renewal_walkforward_{ts}.json')
    json.dump({'csv': os.path.basename(csv), 'extra': extra and os.path.basename(extra),
               'summary': summary, 'folds': folds, 'params': params,
               'ablate': bool(args.ablate)}, open(out, 'w'), indent=2)
    pool.to_csv(os.path.join(DATA_DIR, f'renewal_wf_pool_{ts}.csv'), index=False)
    print(f"\n[OK] {out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
