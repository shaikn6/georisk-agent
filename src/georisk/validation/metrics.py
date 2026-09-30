"""Dependency-free evaluation and calibration maths (numpy only)."""

from __future__ import annotations

import numpy as np


def auc(y: np.ndarray, score: np.ndarray) -> float:
    """ROC AUC via the rank-sum identity; ties get average ranks. NaN if one class is missing."""
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    sorted_s = score[order]
    i = 0
    while i < len(score):
        j = i
        while j + 1 < len(score) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def bootstrap_auc_ci(y, score, n: int = 500, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    y, score = np.asarray(y), np.asarray(score)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        a = auc(y[idx], score[idx])
        if not np.isnan(a):
            vals.append(a)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (float("nan"),) * 2


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class Logistic:
    """L2-regularised logistic regression with feature standardisation (Newton/IRLS)."""

    def __init__(self, l2: float = 1.0, iters: int = 50) -> None:
        self.l2, self.iters = l2, iters
        self.mean = self.std = self.w = None
        self.b = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> Logistic:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.mean, self.std = X.mean(0), X.std(0)
        self.std[self.std < 1e-9] = 1.0
        Z = np.hstack([(X - self.mean) / self.std, np.ones((len(X), 1))])
        w = np.zeros(Z.shape[1])
        reg = np.eye(Z.shape[1]) * self.l2
        reg[-1, -1] = 0.0  # never penalise the intercept
        for _ in range(self.iters):
            p = sigmoid(Z @ w)
            grad = Z.T @ (p - y) + reg @ w
            hess = (Z * (p * (1 - p))[:, None]).T @ Z + reg + 1e-6 * np.eye(Z.shape[1])
            step = np.linalg.solve(hess, grad)
            w -= step
            if np.abs(step).max() < 1e-8:
                break
        self.w, self.b = w[:-1], float(w[-1])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Z = (np.asarray(X, dtype=float) - self.mean) / self.std
        return sigmoid(Z @ self.w + self.b)

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist(), "w": self.w.tolist(), "b": self.b, "l2": self.l2}

    @classmethod
    def from_dict(cls, d: dict) -> Logistic:
        m = cls(l2=d.get("l2", 1.0))
        m.mean, m.std, m.w, m.b = np.array(d["mean"]), np.array(d["std"]), np.array(d["w"]), d["b"]
        return m


def stratified_kfold(y: np.ndarray, k: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    folds = [[] for _ in range(k)]
    for cls in (0, 1):
        idx = rng.permutation(np.where(y == cls)[0])
        for i, j in enumerate(idx):
            folds[i % k].append(j)
    for i in range(k):
        test = np.array(sorted(folds[i]))
        train = np.array(sorted(set(range(len(y))) - set(test.tolist())))
        yield train, test


def cross_validate(X: np.ndarray, y: np.ndarray, k: int = 5, l2: float = 1.0, seed: int = 0) -> dict:
    """Out-of-fold predictions -> honest AUC / Brier (every sample scored by a model that never saw it)."""
    X, y = np.asarray(X, dtype=float), np.asarray(y).astype(int)
    oof = np.zeros(len(y))
    for tr, te in stratified_kfold(y, k, seed):
        oof[te] = Logistic(l2).fit(X[tr], y[tr]).predict_proba(X[te])
    lo, hi = bootstrap_auc_ci(y, oof, seed=seed)
    return {"auc": auc(y, oof), "auc_ci95": [lo, hi], "brier": brier(y, oof),
            "brier_baseline": brier(y, np.full(len(y), y.mean())), "oof": oof}


def leave_group_out(X: np.ndarray, y: np.ndarray, groups, l2: float = 1.0, seed: int = 0) -> dict:
    """Train on every region except one, predict the held-out region. Tests generalisation to places the
    model has never seen, which is what a deployed service actually faces (random K-fold leaks spatial
    autocorrelation between neighbouring samples)."""
    X, y, groups = np.asarray(X, dtype=float), np.asarray(y).astype(int), np.asarray(groups)
    oof = np.full(len(y), np.nan)
    for g in np.unique(groups):
        te = groups == g
        tr = ~te
        if len(np.unique(y[tr])) < 2:
            continue
        oof[te] = Logistic(l2).fit(X[tr], y[tr]).predict_proba(X[te])
    ok = ~np.isnan(oof)
    lo, hi = bootstrap_auc_ci(y[ok], oof[ok], seed=seed)
    per_group = {str(g): auc(y[groups == g], oof[groups == g]) for g in np.unique(groups)
                 if len(np.unique(y[groups == g])) == 2}
    return {"auc": auc(y[ok], oof[ok]), "auc_ci95": [lo, hi], "brier": brier(y[ok], oof[ok]),
            "brier_baseline": brier(y[ok], np.full(ok.sum(), y[ok].mean())), "n": int(ok.sum()),
            "per_group_auc": per_group, "oof": oof}
