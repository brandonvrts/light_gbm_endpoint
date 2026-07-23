"""Duration model v0.1: predict REMAINING tenure (months, capped at 48) for each active
resident at a point in time — the second circle of the two-model Venn (churn risk x
predicted stay length).

Design (2026-07-22, per Brandon):
  * Label: months from the anchor date until the resident's lineage-aware move-out
    (unit transfers chain, so a transfer is NOT the end of a tenancy), capped at 48.
    Anchors are chosen so every label is fully observable under the cap — training
    anchors 2016-01..2021-07 (semiannual) and the FROZEN backtest anchor 2022-01-01
    all have >=54 months of subsequent history, so "still here" simply = 48+.
  * Features: the same point-in-time frame the churn model serves (lease timing incl.
    LeaseTermMonths/DaysToLeaseEnd per the "take lease duration into consideration"
    requirement), MINUS the anchor-clock/era columns (DecisionYear/Month,
    MoveInYear/Month) that would let the model memorize which era an anchor came from.
  * Eval: NO walk-forward — one frozen prediction at 2022-01-01 scored against what
    actually happened through today (the requested backtest), vs naive baselines.
  * Output: bundles/multifamily_duration/v0.1-jan22-backtest/ (model + manifest).

Run: PYTHONPATH=. .venv/bin/python scripts/train_multifamily_duration.py
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr

from core import db
from contracts.multifamily_churn import data

CAP_MONTHS = 48.0
DAYS_PER_MONTH = 30.44
TRAIN_ANCHORS = [f"{y}-{m:02d}-01" for y in range(2016, 2022) for m in (1, 7)]
BACKTEST_ANCHOR = "2022-01-01"
ERA_COLS = {"MoveInYear", "MoveInMonth", "DecisionYear", "DecisionMonth"}
# the lease CLOCK + notice: legitimate point-in-time features, but they let the model
# answer "when does the current lease end" instead of "what kind of stayer is this" —
# the trait variant (the Venn's second circle) excludes them
CLOCK_COLS = {"DaysToLeaseEnd", "IsLeaseExpiringSoon", "HasNotice", "DaysSinceNotice"}
BASE = [c for c in data.MODEL_FEATURES
        if c not in ERA_COLS and c not in ("PropertyName_encoded", "PrimaryPetType_encoded")]
BASE += ["PropertyId"]              # property as a plain numeric categorical
VARIANTS = {
    "full": BASE,                                        # timing-aware (best pure forecast)
    "trait": [c for c in BASE if c not in CLOCK_COLS],   # stayer-type (the Venn circle)
}
BUNDLE_DIR = ROOT / "bundles" / "multifamily_duration" / "v0.1-jan22-backtest"


# ---------------------------------------------------------------- lineage move-outs
def _lineage_ends() -> pd.DataFrame:
    """Per TENANT record: the lineage-aware final move-out (unit transfers chained by
    the same name-or-email + <=31-day gap + same property rule the churn machinery
    uses). Returns SCODE -> (HMYPERSON, lineage_end: date|NaT)."""
    df = db.query(f"""
        SELECT LTRIM(RTRIM(SCODE)) AS SCODE, HMYPERSON, HPROPERTY, DTMOVEIN, DTMOVEOUT,
               UPPER(LTRIM(RTRIM(ISNULL(SFIRSTNAME,''))) + ' ' +
                     LTRIM(RTRIM(ISNULL(SLASTNAME,'')))) AS NM,
               LOWER(LTRIM(RTRIM(ISNULL(SEMAIL,'')))) AS EM
        FROM [{db.DB_YARDI}].dbo.TENANT
        WHERE HPROPERTY IN ({','.join(str(p) for p in data.DEFAULT_PROPERTY_IDS)})
          AND DTMOVEIN IS NOT NULL""")
    df["DTMOVEIN"] = pd.to_datetime(df["DTMOVEIN"])
    df["DTMOVEOUT"] = pd.to_datetime(df["DTMOVEOUT"])
    recs = df.to_dict("records")
    # successor: same property, (name match OR non-empty email match), move-in within
    # 31 days after this record's move-out — indexed by (prop, name)/(prop, email)
    by_nm: dict = {}
    by_em: dict = {}
    for r in recs:
        if r["NM"]:
            by_nm.setdefault((r["HPROPERTY"], r["NM"]), []).append(r)
        if r["EM"]:
            by_em.setdefault((r["HPROPERTY"], r["EM"]), []).append(r)
    succ: dict[int, dict] = {}
    for r in recs:
        mo = r["DTMOVEOUT"]
        if pd.isna(mo):
            continue
        cands = []
        if r["NM"]:
            cands += by_nm.get((r["HPROPERTY"], r["NM"]), [])
        if r["EM"]:
            cands += by_em.get((r["HPROPERTY"], r["EM"]), [])
        best = None
        for c in cands:
            if c is r or pd.isna(c["DTMOVEIN"]):
                continue
            gap = (c["DTMOVEIN"] - mo).days
            if 0 <= gap <= 31 and (best is None or c["DTMOVEIN"] < best["DTMOVEIN"]):
                best = c
        if best is not None:
            succ[int(r["HMYPERSON"])] = best
    ends = {}
    for r in recs:
        cur, seen = r, set()
        while True:
            hmy = int(cur["HMYPERSON"])
            if hmy in seen:            # cycle guard
                break
            seen.add(hmy)
            nxt = succ.get(hmy)
            if nxt is None or pd.isna(cur["DTMOVEOUT"]):
                break
            cur = nxt
        ends[r["SCODE"]] = (int(r["HMYPERSON"]), cur["DTMOVEOUT"],
                            int(cur["HMYPERSON"]))
    out = pd.DataFrame([{"SCODE": k, "HMYPERSON": v[0], "lineage_end": v[1],
                         "lineage_terminal": v[2]} for k, v in ends.items()])
    return out


def _panel(anchor: str, ends: pd.DataFrame) -> pd.DataFrame:
    scodes = data.active_rentable_scodes(None, anchor)
    if not scodes:
        return pd.DataFrame()
    df = data.cohort_features_asof(scodes, anchor)
    df = data.apply_semantic_fixes(df, anchor)
    df["SCODE"] = df["SCODE"].astype(str).str.strip()
    df = df.merge(ends, on="SCODE", how="left")
    an = pd.Timestamp(anchor)
    rem = (df["lineage_end"] - an).dt.days / DAYS_PER_MONTH
    df["remaining_months"] = rem.clip(upper=CAP_MONTHS).where(df["lineage_end"].notna(),
                                                              CAP_MONTHS)
    # guard: a lineage_end BEFORE the anchor means identity chaining glitched — drop
    df = df[(df["remaining_months"] >= 0)]
    df["anchor"] = anchor
    return df


def _matrix(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for c in features:
        X[c] = pd.to_numeric(df.get(c), errors="coerce")
    return X


def _bucket(v: float) -> str:
    if v < 12: return "<12mo"
    if v < 24: return "12-24mo"
    if v < 48: return "24-48mo"
    return "48+mo"


def main() -> None:
    print("building lineage move-outs...")
    ends = _lineage_ends()
    print(f"  {len(ends)} tenancy records chained")

    frames = []
    for a in TRAIN_ANCHORS:
        p = _panel(a, ends)
        print(f"  anchor {a}: {len(p)} residents")
        frames.append(p)
    train = pd.concat(frames, ignore_index=True)
    print(f"train panel: {len(train)} rows, {train['lineage_terminal'].nunique()} distinct lineages")

    bt = _panel(BACKTEST_ANCHOR, ends)
    print(f"backtest panel {BACKTEST_ANCHOR}: {len(bt)} residents")

    ytr = train["remaining_months"].to_numpy()
    ybt = bt["remaining_months"].to_numpy()
    tenure = pd.to_numeric(bt["TenureMonths"], errors="coerce").fillna(0).to_numpy()
    base_mean = float(np.mean(np.abs(ytr.mean() - ybt)))
    rho_tenure = float(spearmanr(tenure, ybt).statistic)

    models, evals = {}, {}
    for tag, feats in VARIANTS.items():
        model = LGBMRegressor(objective="l1", n_estimators=700, learning_rate=0.05,
                              num_leaves=31, min_child_samples=30, subsample=0.9,
                              colsample_bytree=0.9, random_state=7, verbose=-1)
        model.fit(_matrix(train, feats), ytr)
        pred = np.clip(model.predict(_matrix(bt, feats)), 0, CAP_MONTHS)
        mae = float(np.mean(np.abs(pred - ybt)))
        med = float(np.median(np.abs(pred - ybt)))
        rho = float(spearmanr(pred, ybt).statistic)
        r2 = 1 - float(np.sum((ybt - pred) ** 2)) / float(np.sum((ybt - ybt.mean()) ** 2))
        q = pd.qcut(pred, 10, labels=False, duplicates="drop")
        cal = pd.DataFrame({"pred": pred, "real": ybt, "q": q}).groupby("q").agg(
            pred_mean=("pred", "mean"), real_mean=("real", "mean"), n=("pred", "size"))
        bp = pd.Series([_bucket(v) for v in pred]); br = pd.Series([_bucket(v) for v in ybt])
        conf = pd.crosstab(bp, br, rownames=["predicted"], colnames=["realized"])
        imp = sorted(zip(feats, model.feature_importances_), key=lambda kv: -kv[1])[:12]

        print(f"\n== FROZEN BACKTEST · {tag.upper()} variant "
              f"(trained <=2021-07, predicted {BACKTEST_ANCHOR}, realized through today) ==")
        print(f"n={len(bt)}  MAE={mae:.1f}mo  medianAE={med:.1f}mo  Spearman={rho:.3f}  R2={r2:.3f}")
        print(f"baselines: predict-train-mean MAE={base_mean:.1f}mo · tenure-as-ranker "
              f"Spearman={rho_tenure:.3f}")
        print("decile calibration (predicted vs realized mean months):")
        print(cal.round(1).to_string())
        print("bucket matrix (rows=predicted, cols=realized):")
        print(conf.to_string())
        print("top features:")
        for f, v in imp:
            print(f"  {f:26s} {v}")

        models[tag] = {"model": model, "features": feats}
        evals[tag] = {
            "features_n": len(feats),
            "mae_months": round(mae, 2), "median_ae_months": round(med, 2),
            "spearman": round(rho, 3), "r2": round(r2, 3),
            "decile_calibration": [
                {"pred_mean": round(float(r.pred_mean), 1),
                 "real_mean": round(float(r.real_mean), 1), "n": int(r.n)}
                for r in cal.itertuples()],
            "bucket_matrix": {str(i): {str(c): int(conf.loc[i, c]) for c in conf.columns}
                              for i in conf.index},
            "feature_importance_top": [{"feature": f, "gain": int(v)} for f, v in imp]}

    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"variants": models, "cap_months": CAP_MONTHS}, BUNDLE_DIR / "model.joblib")
    manifest = {
        "contract": "multifamily-duration", "version": "v0.1-jan22-backtest",
        "status": "INTERNAL (report-only, not served)",
        "target": f"remaining tenure in months from anchor, capped at {CAP_MONTHS:.0f} "
                  f"(fully observable for all anchors used — no censoring)",
        "algorithm": "lightgbm l1 regression",
        "variants": {"full": "timing-aware forecast (lease clock + notice included)",
                     "trait": "stayer-type score (lease clock + notice EXCLUDED — the "
                              "second circle of the two-model Venn; rank-use only)"},
        "clock_columns_excluded_from_trait": sorted(CLOCK_COLS),
        "era_columns_excluded": sorted(ERA_COLS),
        "train_anchors": TRAIN_ANCHORS,
        "train_rows": int(len(train)), "train_lineages": int(train["lineage_terminal"].nunique()),
        "labels": "lineage-aware move-out (unit transfers chained: same property + "
                  "name-or-email + <=31d gap)",
        "backtest": {"anchor": BACKTEST_ANCHOR, "n": int(len(bt)),
                     "baseline_train_mean_mae": round(base_mean, 2),
                     "baseline_tenure_spearman": round(rho_tenure, 3), **evals},
        "regime_caveat": "the 2022 cohort was tenure-bimodal (post-COVID pile-up, then the "
                         "2022 purge) and thus unusually separable; today's population has "
                         "more mid-tenure mass — treat absolute months as rank scores",
        "trained_at": date.today().isoformat(),
    }
    (BUNDLE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"\nsaved bundle -> {BUNDLE_DIR}")


if __name__ == "__main__":
    main()
