"""Vendored calibration transforms for churn bundles (unpickled at serve time; kept at
ROOT alongside platt_rescaler.py so registry.load's sys.path covers them).

SplineCalibrator (v6+): a smooth, flexible, MONOTONE probability calibrator. It replaces
v5's single global sigmoid + intercept, which was too stiff in the mid-range (deciles
5-8 predicted ~1.5-2x low) and piled 70%+ of scores on the floor. Isotonic regression is
flexible but piecewise-CONSTANT -> it reintroduces the snapped-score problem v5 was built
to kill. This calibrator gets both: it fits the empirical churn rate per logit-quantile
bin (flexible), isotonic-smooths those bin rates (monotone), then interpolates them with a
monotone cubic spline (PCHIP) so the output is smooth and strictly rank-preserving. A
separate additive logit `intercept` carries the expiring-cohort population-shift level
correction (fit downstream on the report cohort).
"""
from __future__ import annotations

import numpy as np


def _logit(p, eps: float = 1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


class SplineCalibrator:
    """Wraps a fitted base classifier. predict_proba(X_scaled) -> calibrated [P0, P1]."""

    def __init__(self, inner_model, knots_z, knots_zc, intercept: float = 0.0,
                 floor: float = 0.005, ceil: float = 0.95):
        self.inner_model = inner_model
        self.knots_z = np.asarray(knots_z, dtype=float)
        self.knots_zc = np.asarray(knots_zc, dtype=float)
        self.intercept = float(intercept)
        self.floor = float(floor)
        self.ceil = float(ceil)

    # ---- fitting -----------------------------------------------------------
    @staticmethod
    def fit(inner_model, X_scaled, y, n_knots: int = 20, floor: float = 0.005,
            ceil: float = 0.95) -> "SplineCalibrator":
        from sklearn.isotonic import IsotonicRegression

        raw = np.asarray(inner_model.predict_proba(X_scaled))[:, 1]
        z = _logit(raw)
        y = np.asarray(y, dtype=float)
        # quantile knots on the logit; dedupe collapsed edges
        qs = np.linspace(0, 1, n_knots + 1)
        edges = np.unique(np.quantile(z, qs))
        if len(edges) < 3:                       # degenerate: fall back to identity knots
            edges = np.unique(np.quantile(z, [0, 0.5, 1.0]))
        idx = np.clip(np.digitize(z, edges[1:-1]), 0, len(edges) - 2)
        knot_z, knot_rate = [], []
        for b in range(len(edges) - 1):
            m = idx == b
            if m.sum() < 5:
                continue
            knot_z.append(float(z[m].mean()))
            knot_rate.append(float(y[m].mean()))
        knot_z = np.asarray(knot_z)
        knot_rate = np.asarray(knot_rate)
        # isotonic-smooth the empirical rates so the spline is monotone non-decreasing
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        knot_rate = iso.fit_transform(knot_z, knot_rate)
        knots_zc = _logit(knot_rate)
        # ensure strictly increasing x for PCHIP
        order = np.argsort(knot_z)
        knot_z, knots_zc = knot_z[order], knots_zc[order]
        keep = np.concatenate([[True], np.diff(knot_z) > 1e-9])
        return SplineCalibrator(inner_model, knot_z[keep], knots_zc[keep],
                                intercept=0.0, floor=floor, ceil=ceil)

    def with_intercept(self, intercept: float) -> "SplineCalibrator":
        return SplineCalibrator(self.inner_model, self.knots_z, self.knots_zc,
                                intercept=intercept, floor=self.floor, ceil=self.ceil)

    # ---- transform ---------------------------------------------------------
    def _interp(self, z):
        from scipy.interpolate import PchipInterpolator
        z = np.asarray(z, dtype=float)
        if len(self.knots_z) < 2:                # degenerate calibrator -> identity
            return z
        pc = PchipInterpolator(self.knots_z, self.knots_zc, extrapolate=False)
        out = pc(z)
        lo, hi = self.knots_z[0], self.knots_z[-1]
        d = pc.derivative()
        below, above = z < lo, z > hi
        # linear extrapolation with the (non-negative) endpoint slope: stays monotone,
        # preserves ranking of the most extreme residents instead of clamping them to ties
        out[below] = self.knots_zc[0] + max(0.0, float(d(lo))) * (z[below] - lo)
        out[above] = self.knots_zc[-1] + max(0.0, float(d(hi))) * (z[above] - hi)
        return out

    def predict_proba(self, X):
        raw = np.asarray(self.inner_model.predict_proba(X))[:, 1]
        zc = self._interp(_logit(raw)) + self.intercept
        p = 1.0 / (1.0 + np.exp(-zc))
        p = np.clip(p, self.floor, self.ceil)
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
