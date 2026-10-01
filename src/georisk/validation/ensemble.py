"""Independent model families combined by averaging (numpy only, JSON-serialisable).

Each member has a different inductive bias, so their errors are only partly correlated: a linear model
(Logistic), a bagged forest of shallow trees (interactions/thresholds) and a k-nearest-neighbour model
(local structure). Averaging probabilities reduces variance, and one member misbehaving cannot dominate.
"""

from __future__ import annotations

import numpy as np

from .metrics import Logistic


class Forest:
    """Bagged depth-limited CART trees with random feature subsets; probability = mean leaf positive rate."""

    def __init__(self, n_trees: int = 60, depth: int = 3, min_leaf: int = 5, seed: int = 0) -> None:
        self.n_trees, self.depth, self.min_leaf, self.seed = n_trees, depth, min_leaf, seed
        self.trees: list = []

    def _grow(self, X, y, rng, d):
        p = float((y.sum() + 1) / (len(y) + 2))  # Laplace-smoothed leaf probability
        if d == 0 or len(y) < 2 * self.min_leaf or y.min() == y.max():
            return p
        m = max(1, int(np.sqrt(X.shape[1])))
        best = None
        for j in rng.choice(X.shape[1], m, replace=False):
            for t in np.unique(np.quantile(X[:, j], [0.2, 0.35, 0.5, 0.65, 0.8])):
                left = X[:, j] <= t
                nl, nr = left.sum(), (~left).sum()
                if nl < self.min_leaf or nr < self.min_leaf:
                    continue
                gini = nl * _gini(y[left]) + nr * _gini(y[~left])
                if best is None or gini < best[0]:
                    best = (gini, int(j), float(t), left)
        if best is None:
            return p
        _, j, t, left = best
        return [j, t, self._grow(X[left], y[left], rng, d - 1), self._grow(X[~left], y[~left], rng, d - 1)]

    def fit(self, X, y) -> Forest:
        X, y = np.asarray(X, float), np.asarray(y, int)
        rng = np.random.default_rng(self.seed)
        self.trees = []
        for _ in range(self.n_trees):
            idx = rng.integers(0, len(y), len(y))
            self.trees.append(self._grow(X[idx], y[idx], rng, self.depth))
        return self

    @staticmethod
    def _walk(node, x):
        while isinstance(node, list):
            node = node[2] if x[node[0]] <= node[1] else node[3]
        return node

    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X, float)
        return np.array([np.mean([self._walk(t, x) for t in self.trees]) for x in X])

    def to_dict(self) -> dict:
        return {"trees": self.trees}

    @classmethod
    def from_dict(cls, d: dict) -> Forest:
        f = cls()
        f.trees = d["trees"]
        return f


def _gini(y) -> float:
    p = y.mean()
    return 2 * p * (1 - p)


class SkForest(Forest):
    """Trained with scikit-learn (research extra) but exported to the same JSON trees, so serving needs numpy only."""

    def __init__(self, n_trees: int = 200, min_leaf: int = 3, seed: int = 0) -> None:
        super().__init__(n_trees=n_trees, min_leaf=min_leaf, seed=seed)

    def fit(self, X, y) -> SkForest:
        from sklearn.ensemble import RandomForestClassifier

        rf = RandomForestClassifier(self.n_trees, min_samples_leaf=self.min_leaf, random_state=self.seed).fit(X, y)
        pos = list(rf.classes_).index(1)
        self.trees = [_export_tree(e.tree_, 0, pos) for e in rf.estimators_]
        return self


def _export_tree(t, node: int, pos: int):
    if t.children_left[node] == -1:
        v = t.value[node][0]
        return float((v[pos] + 0.5) / (v.sum() + 1.0))  # smoothed leaf probability
    return [int(t.feature[node]), float(t.threshold[node]),
            _export_tree(t, int(t.children_left[node]), pos), _export_tree(t, int(t.children_right[node]), pos)]


class KNN:
    """Distance-weighted k-nearest neighbours in standardised feature space."""

    def __init__(self, k: int = 9) -> None:
        self.k = k
        self.mean = self.std = self.X = self.y = None

    def fit(self, X, y) -> KNN:
        X = np.asarray(X, float)
        self.mean, self.std = X.mean(0), X.std(0)
        self.std[self.std < 1e-9] = 1.0
        self.X, self.y = (X - self.mean) / self.std, np.asarray(y, float)
        return self

    def predict_proba(self, X) -> np.ndarray:
        Z = (np.asarray(X, float) - self.mean) / self.std
        out = []
        for z in Z:
            d = np.sqrt(((self.X - z) ** 2).sum(1))
            idx = np.argsort(d)[: min(self.k, len(d))]
            w = 1.0 / (d[idx] + 1e-3)
            out.append(float((w * self.y[idx]).sum() / w.sum()))
        return np.clip(np.array(out), 0.02, 0.98)

    def to_dict(self) -> dict:
        return {"k": self.k, "mean": self.mean.tolist(), "std": self.std.tolist(),
                "X": self.X.round(4).tolist(), "y": self.y.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> KNN:
        m = cls(d["k"])
        m.mean, m.std, m.X, m.y = np.array(d["mean"]), np.array(d["std"]), np.array(d["X"]), np.array(d["y"])
        return m


ZMAX = 2.0  # held-out tuning (z-clip 2.0 beat 3.0 and uncapped): AUC +0.008, Brier -0.003
MEMBERS = {"logistic": Logistic, "forest": Forest, "sk_forest": SkForest, "knn": KNN}


class Ensemble:
    def __init__(self, l2: float = 3.0, names: tuple[str, ...] = ("logistic", "forest", "knn")) -> None:
        self.l2, self.names = l2, names
        self.models: dict = {}

    def fit(self, X, y) -> Ensemble:
        self.models = {n: (Logistic(self.l2, zmax=ZMAX) if n == "logistic" else MEMBERS[n]()).fit(X, y) for n in self.names}
        return self

    def member_probas(self, X) -> dict[str, np.ndarray]:
        return {n: m.predict_proba(X) for n, m in self.models.items()}

    def predict_proba(self, X) -> np.ndarray:
        return np.mean(list(self.member_probas(X).values()), axis=0)

    def to_dict(self) -> dict:
        return {"type": "ensemble", "members": {n: m.to_dict() for n, m in self.models.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> Ensemble:
        e = cls()
        e.models = {n: MEMBERS[n].from_dict(v) for n, v in d["members"].items()}
        return e


def dump_model(name: str, model) -> dict:
    """Serialise any served model with its family tag so it reloads as the right class."""
    return {"type": name, **model.to_dict()}


def load_model(d: dict):
    """Dispatch on the family tag; untagged artifacts (before v0.3) are bare Logistic dicts."""
    kind = d.get("type", "logistic")
    return Ensemble.from_dict(d) if kind in ("ensemble", "blend") else MEMBERS[kind].from_dict(d)
