"""Minimal Firth penalized-likelihood logistic regression (+ FLIC intercept
correction), vendored at project root so served bundles can unpickle it.

Firth (Heinze & Schemper 2002) reduces small-sample / rare-event bias and handles
separation. FLIC (Puhr et al. 2017) re-estimates the intercept so the mean predicted
probability matches the observed event rate, removing Firth's upward probability bias
at low prevalence. Self-contained (standardizes internally) and sklearn-shaped
(fit / predict_proba) so it slots into the bundle + contract like any classifier.
"""
from __future__ import annotations
import numpy as np


class FirthLogisticRegression:
    def __init__(self, max_iter: int = 200, tol: float = 1e-6, flic: bool = True):
        self.max_iter = max_iter
        self.tol = tol
        self.flic = flic
        self.classes_ = np.array([0, 1])

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        Xs = (X - self.mean_) / self.std_
        Xd = np.column_stack([np.ones(len(Xs)), Xs])      # [intercept | scaled features]
        beta = np.zeros(Xd.shape[1])
        for _ in range(self.max_iter):
            p = self._sigmoid(Xd @ beta)
            W = np.clip(p * (1 - p), 1e-10, None)
            XtWX = Xd.T @ (Xd * W[:, None])
            XtWX_inv = np.linalg.pinv(XtWX)
            # hat diagonal h_i = W_i * x_i' (X'WX)^-1 x_i
            hdiag = W * np.einsum("ij,jk,ik->i", Xd, XtWX_inv, Xd)
            # Firth-modified score: U*_j = sum_i (y_i - p_i + h_i(0.5 - p_i)) x_ij
            U = Xd.T @ (y - p + hdiag * (0.5 - p))
            step = XtWX_inv @ U
            beta = beta + step
            if np.max(np.abs(step)) < self.tol:
                break
        if self.flic:
            # FLIC: keep Firth slopes, re-estimate intercept by plain MLE with the
            # linear predictor as an offset -> mean(p) == mean(y), fixing the level.
            offset = Xs @ beta[1:]
            ybar = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
            a = np.log(ybar / (1 - ybar))
            for _ in range(100):
                p = self._sigmoid(a + offset)
                g = np.sum(y - p)
                h = -np.sum(np.clip(p * (1 - p), 1e-12, None))
                a_new = a - g / h
                if abs(a_new - a) < 1e-9:
                    a = a_new
                    break
                a = a_new
            beta = beta.copy()
            beta[0] = a
        self.coef_ = beta
        return self

    def decision_function(self, X):
        Xs = (np.asarray(X, dtype=float) - self.mean_) / self.std_
        Xd = np.column_stack([np.ones(len(Xs)), Xs])
        return Xd @ self.coef_

    def predict_proba(self, X):
        p = self._sigmoid(self.decision_function(X))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
