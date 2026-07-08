"""
Standalone module for the PlattRescaler wrapper used by recalibrate_h180.

Keeping this in its own module (not under __main__) so joblib's pickle
preserves the import path -- otherwise consumers that just load the
bundle (executive_brief.py, validate_*.py) would fail with
AttributeError: module '__main__' has no attribute 'PlattRescaler'.
"""

from __future__ import annotations

import numpy as np


class PlattRescaler:
    """A 1-parameter logistic rescaling layer applied on top of an
    already-calibrated probabilistic model.

    new_p = sigmoid(a + b * logit(old_p))

    The (a, b) parameters are fit on holdout data by recalibrate_h180.py.
    `predict_proba` returns the same (n, 2) shape the wrapped model does
    so consumer code doesn't need to know it's been wrapped.
    """

    def __init__(self, inner_model, a: float, b: float):
        self.inner_model = inner_model
        self.a = float(a)
        self.b = float(b)

    @staticmethod
    def _logit(p, eps: float = 1e-6):
        p = np.clip(p, eps, 1 - eps)
        return np.log(p / (1 - p))

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    def predict_proba(self, X):
        raw = self.inner_model.predict_proba(X)
        p_old = raw[:, 1]
        z = self.a + self.b * self._logit(p_old)
        p_new = self._sigmoid(z)
        out = np.empty_like(raw)
        out[:, 0] = 1.0 - p_new
        out[:, 1] = p_new
        return out

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
