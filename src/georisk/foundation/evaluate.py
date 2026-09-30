"""Do Prithvi embeddings help? Same leave-region-out protocol as calibrate.py, so the numbers are directly comparable.

``python -m georisk.foundation.evaluate`` -> validation_data/foundation_report.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..validation.calibrate import L2, TASKS, _lro, build_matrix, load_rows
from ..validation.dataset import OHIO
from ..validation.ensemble import Ensemble, SkForest
from ..validation.metrics import Logistic, auc, bootstrap_auc_ci, brier
from .chips import key

EMB = Path("validation_data/prithvi_embeddings.npz")
FLOOD = [Path(f"validation_data/{n}.jsonl") for n in ("flood", "flood2", "flood3", "flood_ohio")]
FIRE = [Path(f"validation_data/{n}.jsonl") for n in ("fire", "fire2")]


class EmbModel:
    """PCA of the embedding columns (fitted on the training fold only), optionally joined with the tabular features."""

    def __init__(self, n_feat: int, k: int = 32, base: str = "logistic", use_feat: bool = False) -> None:
        self.n_feat, self.k, self.base, self.use_feat = n_feat, k, base, use_feat

    def _design(self, X: np.ndarray) -> np.ndarray:
        P = ((X[:, self.n_feat :] - self.mu) / self.sd) @ self.V
        return np.hstack([X[:, : self.n_feat], P]) if self.use_feat else P

    def fit(self, X: np.ndarray, y: np.ndarray) -> EmbModel:
        Z = X[:, self.n_feat :]
        self.mu, self.sd = Z.mean(0), Z.std(0) + 1e-6
        _, _, vt = np.linalg.svd((Z - self.mu) / self.sd, full_matrices=False)
        self.V = vt[: self.k].T
        self.model = (Logistic(L2 * 3) if self.base == "logistic" else SkForest()).fit(self._design(X), y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self._design(X))


def _row(y, p) -> dict:
    lo, hi = bootstrap_auc_ci(y, p)
    return {"auc": round(auc(y, p), 3), "auc_ci95": [round(lo, 3), round(hi, 3)], "brier": round(brier(y, p), 4)}


def evaluate(task: str, files: list[Path]) -> dict:
    data = np.load(EMB)
    index = {k: i for i, k in enumerate(data["keys"])}
    rows = [r for r in load_rows(*files) if key(r["lat"], r["lon"]) in index]
    X, y, _ = build_matrix(rows, TASKS[task]["features"])
    Z = np.stack([data["emb"][index[key(r["lat"], r["lon"])]] for r in rows]).astype("float64")
    groups = np.array([r["region"] for r in rows])
    n = X.shape[1]
    XZ = np.hstack([X, Z])
    cands = {
        "features: logistic": (lambda: Logistic(L2), X),
        "features: forest": (SkForest, X),
        "features: blend (logistic + forest)": (lambda: Ensemble(L2, ("logistic", "sk_forest")), X),
        "prithvi only: logistic on PCA32": (lambda: EmbModel(n, 32, "logistic"), XZ),
        "prithvi only: forest on PCA32": (lambda: EmbModel(n, 32, "forest"), XZ),
        "features + prithvi PCA16: logistic": (lambda: EmbModel(n, 16, "logistic", True), XZ),
        "features + prithvi PCA16: forest": (lambda: EmbModel(n, 16, "forest", True), XZ),
    }
    res, oofs = {}, {}
    for name, (mk, M) in cands.items():
        oofs[name], res[name] = _lro(mk, M, y, groups)
    ok = ~np.isnan(oofs["features: blend (logistic + forest)"])
    avg = (oofs["features: blend (logistic + forest)"][ok] + oofs["prithvi only: logistic on PCA32"][ok]) / 2
    res["average: feature blend + prithvi logistic"] = _row(y[ok], avg)
    best = max(res, key=lambda k: res[k]["auc"])
    out = {"task": task, "n": len(rows), "missing_chips": len(load_rows(*files)) - len(rows), "leave_region_out": res, "best": best}
    if task == "flood":
        oh = np.array([r["region"] in OHIO for r in rows])
        ohio = {}
        for name, (mk, M) in cands.items():
            if name.startswith(("features: blend", "prithvi only: logistic", "features + prithvi PCA16: forest")):
                ohio[name] = _row(y[oh], mk().fit(M[~oh], y[~oh]).predict_proba(M[oh]))
        out["ohio_holdout"] = {"n_ohio": int(oh.sum()), "by_model": ohio}
    return out


def main() -> None:
    reports = [evaluate("flood", FLOOD), evaluate("fire", FIRE)]
    Path("validation_data/foundation_report.json").write_text(json.dumps(reports, indent=2))
    for r in reports:
        print(f"\n{r['task']}  n={r['n']} (missing chips: {r['missing_chips']})  best: {r['best']}")
        for k, v in sorted(r["leave_region_out"].items(), key=lambda kv: -kv[1]["auc"]):
            print(f"  {k:45s} AUC {v['auc']:.3f} {v['auc_ci95']} brier {v['brier']}")
        if "ohio_holdout" in r:
            print("  Ohio holdout:", {k: v["auc"] for k, v in r["ohio_holdout"]["by_model"].items()})


if __name__ == "__main__":
    main()
