"""Minimal L2-regularized logistic regression (pure Python, stdlib only).

Kept dependency-free so the screening logic runs anywhere instantly. For
production, swap in scikit-learn / xgboost via the `ml` extra -- the interface
(fit / predict_proba / log_loss) is intentionally sklearn-compatible.

Features are standardized internally (z-scored) for stable gradient descent;
coefficients are reported in standardized units, which makes them directly
comparable as effect sizes across stats on different scales.
"""
from __future__ import annotations

import math
import random


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


class LogisticRegression:
    def __init__(self, l2: float = 1.0, lr: float = 0.3, epochs: int = 300,
                 seed: int = 0):
        self.l2 = l2
        self.lr = lr
        self.epochs = epochs
        self.seed = seed
        self.w: list[float] = []
        self.b: float = 0.0
        self._mu: list[float] = []
        self._sd: list[float] = []

    # --- standardization ---
    def _fit_scaler(self, X: list[list[float]]) -> None:
        n, d = len(X), len(X[0])
        self._mu = [0.0] * d
        self._sd = [0.0] * d
        for j in range(d):
            col = [row[j] for row in X]
            mu = sum(col) / n
            var = sum((v - mu) ** 2 for v in col) / max(n - 1, 1)
            self._mu[j] = mu
            self._sd[j] = math.sqrt(var) or 1.0

    def _scale(self, row: list[float]) -> list[float]:
        return [(v - mu) / sd for v, mu, sd in zip(row, self._mu, self._sd)]

    # --- API ---
    def fit(self, X: list[list[float]], y: list[int]) -> "LogisticRegression":
        if not X:
            raise ValueError("empty training set")
        self._fit_scaler(X)
        Xs = [self._scale(r) for r in X]
        n, d = len(Xs), len(Xs[0])
        rng = random.Random(self.seed)
        self.w = [rng.uniform(-0.01, 0.01) for _ in range(d)]
        self.b = 0.0
        for _ in range(self.epochs):
            gw = [0.0] * d
            gb = 0.0
            for xi, yi in zip(Xs, y):
                p = _sigmoid(sum(w * x for w, x in zip(self.w, xi)) + self.b)
                err = p - yi
                for j in range(d):
                    gw[j] += err * xi[j]
                gb += err
            for j in range(d):
                gw[j] = gw[j] / n + self.l2 * self.w[j] / n
                self.w[j] -= self.lr * gw[j]
            self.b -= self.lr * gb / n
        return self

    def predict_proba(self, X: list[list[float]]) -> list[float]:
        return [_sigmoid(sum(w * x for w, x in zip(self.w, self._scale(r))) + self.b)
                for r in X]

    def coef(self) -> list[float]:
        """Standardized coefficients (effect sizes in log-odds per SD)."""
        return list(self.w)

    @staticmethod
    def log_loss(y: list[int], p: list[float], eps: float = 1e-12) -> float:
        s = 0.0
        for yi, pi in zip(y, p):
            pi = min(max(pi, eps), 1 - eps)
            s += -(yi * math.log(pi) + (1 - yi) * math.log(1 - pi))
        return s / len(y)

    @staticmethod
    def brier(y: list[int], p: list[float]) -> float:
        return sum((pi - yi) ** 2 for yi, pi in zip(y, p)) / len(y)
