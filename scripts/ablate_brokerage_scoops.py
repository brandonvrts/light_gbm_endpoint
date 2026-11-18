"""Definitive PAIRED ablation: do point-in-time ZoomInfo scoops (2022-2026 backfill) add
lift to the COMPANY-grain conversion model, holding the base features fixed?

v0.4 only compared its absolute PR-AUC (0.115) to a hardcoded v0.3 reference (0.108) -> the
two CIs overlap, so "within noise" was a soft call. This does the rigorous test: identical
rolling-origin folds + identical base features, one model WITHOUT scoops vs one WITH the
elastic-net-selected scoops, then paired-bootstrap the pooled-OOF PR-AUC DELTA (same resample
indices for both). Scoops earn promotion iff the delta CI lower bound > 0.

Reuses the v0.3 harness (same small-n stack: penalized + Firth/FLIC, expanding-window
rolling-origin) and the v0.4 scoop recompute. SELECT-only; writes nothing.

Run: PYTHONPATH=. .venv/bin/python scripts/ablate_brokerage_scoops.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import train_brokerage_v03 as v3                      # noqa: E402
import train_brokerage_v04 as v4                      # noqa: E402
from firth_logistic import FirthLogisticRegression    # noqa: E402

RNG = np.random.default_rng(7)


def paired_delta(y, p_wo, p_w, n=3000):
    """Bootstrap AP(with) - AP(without) on the SAME resample indices."""
    d = []
    for _ in range(n):
        idx = RNG.integers(0, len(y), len(y))
        if y[idx].sum() < 2:
            continue
        d.append(average_precision_score(y[idx], p_w[idx]) - average_precision_score(y[idx], p_wo[idx]))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(np.mean(d)), float(lo), float(hi)


def main():
    print("Loading v0.3 base + recomputing point-in-time scoops (read-only)...")
    df = v3.load_data()
    y = df["Label"].to_numpy().astype(int)
    X_base_all, _ = v3.matrix(df, v3.KEEP)
    scoopX, n_scoops, n_linked = v4.scoop_matrix(df)
    print(f"  rows={len(df)} pos={int(y.sum())} converting_prospects={df.loc[y==1,'hMy'].nunique()} | "
          f"linked_prospects={n_linked} scoop_rows={n_scoops} | "
          f"snapshots w/ >=1 prior scoop={int((scoopX['scoop_cnt_365d']>0).sum())} "
          f"({(scoopX['scoop_cnt_365d']>0).mean():.1%} of rows)")

    # Same elastic-net selection over the augmented pool as v0.4 -> base_sel + scoop_sel
    Xaug = pd.concat([X_base_all, scoopX], axis=1)
    pool = v3.KEEP + v4.SCOOP_FEATS
    Z = StandardScaler().fit_transform(Xaug.values)
    enet = LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, C=0.2,
                              max_iter=5000, random_state=7).fit(Z, y)
    coef = np.abs(enet.coef_[0])
    sel = [pool[i] for i in np.argsort(coef)[::-1] if coef[i] > 1e-8][:10]
    scoop_sel = [f for f in sel if f in v4.SCOOP_FEATS]
    base_sel = [f for f in sel if f not in v4.SCOOP_FEATS]
    print(f"  selected {len(sel)}: base={base_sel}")
    print(f"                       scoops={scoop_sel}")
    if not scoop_sel:
        print("  elastic-net selected NO scoop features -> scoops add nothing by selection. Done.")
        return

    X_wo = Xaug[base_sel]
    X_w = Xaug[base_sel + scoop_sel]

    for name, build in [("penalized-logistic", v3._penalized),
                        ("firth-flic", lambda: FirthLogisticRegression(flic=True))]:
        y_wo, p_wo, _ = v3.rolling_origin(df, X_wo, y, build)
        y_w, p_w, _ = v3.rolling_origin(df, X_w, y, build)
        assert np.array_equal(y_wo, y_w), "fold misalignment"   # same folds/rows -> paired
        ap_wo = average_precision_score(y_wo, p_wo)
        ap_w = average_precision_score(y_w, p_w)
        dmean, dlo, dhi = paired_delta(y_wo, p_wo, p_w)
        verdict = "SCOOPS HELP (CI>0)" if dlo > 0 else ("scoops HURT (CI<0)" if dhi < 0 else "within noise (CI spans 0)")
        print(f"\n[{name}] oof_n={len(y_wo)} pos={int(y_wo.sum())}")
        print(f"   PR-AUC  without={ap_wo:.4f}  with={ap_w:.4f}  point-delta={ap_w-ap_wo:+.4f}")
        print(f"   paired-bootstrap delta mean={dmean:+.4f}  CI95=[{dlo:+.4f}, {dhi:+.4f}]  -> {verdict}")


if __name__ == "__main__":
    main()
