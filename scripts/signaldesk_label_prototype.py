"""READ-ONLY prototype of a CORRECTED, time-localized Signal Desk conversion label,
built from rea.vw_Deal executed-stage events (the only source with a working
prospect link via commProspectId). Label=1 iff the prospect has an Executed deal
within (snapshot, snapshot+180d]. Measures whether walk-forward is feasible.
No DB writes.
"""
from core import db
from contracts.signal_desk_brokerage.schema import FEATURES
import pandas as pd, numpy as np

M = db.DB_MAIN
FCOLS = ", ".join(f"f.[{c}]" for c in FEATURES)

# Corrected label in one query (Executed deal within 180d AFTER the snapshot).
SQL = f"""
SELECT f.hMy, f.SnapshotDate, {FCOLS},
  CASE WHEN EXISTS (
    SELECT 1 FROM [{M}].rea.vw_Deal d
    WHERE d.commProspectId = f.hMy AND d.stageName = 'Executed'
      AND d.stageBeganAt > f.SnapshotDate
      AND d.stageBeganAt <= DATEADD(DAY, 180, f.SnapshotDate)
  ) THEN 1 ELSE 0 END AS Label
FROM [{M}].rea.ProspectFeatureSnapshot f
"""

print("Building corrected label (this is the prototype the owner would productionize)...")
df = db.query(SQL)
print(f"rows: {len(df)} | positives: {int(df['Label'].sum())} | "
      f"distinct positive prospects: {df.loc[df.Label==1,'hMy'].nunique()}")

print("\npositives by quarter:")
df["q"] = pd.to_datetime(df["SnapshotDate"]).dt.to_period("Q").astype(str)
g = df.groupby("q").agg(snaps=("Label", "size"), positives=("Label", "sum"),
                        pos_prospects=("hMy", lambda s: df.loc[s.index][df.loc[s.index, "Label"] == 1]["hMy"].nunique()))
print(g.to_string())

pos_entities = df.loc[df.Label == 1, "hMy"].nunique()
print(f"\n=== VERDICT: {pos_entities} distinct converting prospects (need >=~500 for stable walk-forward) ===")

if pos_entities >= 25:
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = df["Label"].astype(int).to_numpy()
    groups = df["hMy"].to_numpy()
    aucs, prs, briers = [], [], []
    for tr, te in GroupKFold(n_splits=3).split(X, y, groups):
        if y[tr].sum() < 5 or y[te].sum() < 3:
            continue
        m = LGBMClassifier(n_estimators=200, num_leaves=31, learning_rate=0.05,
                           is_unbalance=True, min_child_samples=20, n_jobs=4, verbose=-1)
        m.fit(X.iloc[tr], y[tr])
        p = m.predict_proba(X.iloc[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p)); prs.append(average_precision_score(y[te], p)); briers.append(brier_score_loss(y[te], p))
    if aucs:
        print(f"group-disjoint CV (by prospect): ROC-AUC {np.mean(aucs):.3f}±{np.std(aucs):.3f} | "
              f"PR-AUC {np.mean(prs):.3f} | Brier {np.mean(briers):.3f} | folds {len(aucs)}")
        print("(treat as DIRECTIONAL only; far too few positives for a trustworthy production metric.)")
    else:
        print("Could not form CV folds with enough positives -> NOT walk-forward validatable.")
else:
    print("Too few positives to attempt any CV -> Signal Desk is NOT walk-forward validatable on current data.")
