"""Research benchmark: do heavier scikit-learn models beat the numpy models under leave-region-out?

scikit-learn is a research-only dependency (``pip install '.[research]'``); the served API never imports it.
Usage: ``python -m georisk.validation.benchmark`` -> validation_data/benchmark.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .calibrate import L2, TASKS, build_matrix, load_rows
from .ensemble import Ensemble, Forest
from .metrics import Logistic, auc, bootstrap_auc_ci, brier

FLOOD_FILES = [Path(f"validation_data/{n}.jsonl") for n in ("flood", "flood2", "flood3", "flood_ohio")]
FIRE_FILES = [Path(f"validation_data/{n}.jsonl") for n in ("fire", "fire2")]


def _sk_models() -> dict:
    from sklearn.ensemble import (
        ExtraTreesClassifier,
        HistGradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    return {
        "sk_hist_gbm": lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, random_state=0),
        "sk_random_forest": lambda: RandomForestClassifier(300, min_samples_leaf=3, random_state=0),
        "sk_extra_trees": lambda: ExtraTreesClassifier(300, min_samples_leaf=3, random_state=0),
        "sk_svm_rbf": lambda: make_pipeline(StandardScaler(), SVC(C=1.0, probability=True, random_state=0)),
    }


class _Wrap:
    """Give the numpy models and sklearn models the same fit/predict_proba surface."""

    def __init__(self, m):
        self.m = m

    def fit(self, X, y):
        self.m.fit(X, y)
        return self

    def predict_proba(self, X):
        p = self.m.predict_proba(X)
        return p[:, 1] if p.ndim == 2 else p


def _lro(make, X, y, groups):
    oof = np.full(len(y), np.nan)
    for g in np.unique(groups):
        te = groups == g
        if len(np.unique(y[~te])) == 2:
            oof[te] = make().fit(X[~te], y[~te]).predict_proba(X[te])
    ok = ~np.isnan(oof)
    lo, hi = bootstrap_auc_ci(y[ok], oof[ok])
    return {"auc": round(auc(y[ok], oof[ok]), 3), "ci95": [round(lo, 3), round(hi, 3)], "brier": round(brier(y[ok], oof[ok]), 4)}, oof


def run(task: str, files: list[Path]) -> dict:
    rows = load_rows(*files)
    X, y, _ = build_matrix(rows, TASKS[task]["features"])
    groups = np.array([r["region"] for r in rows])
    models = {"np_logistic": lambda: Logistic(L2), "np_forest": Forest, "np_ensemble": lambda: Ensemble(L2)}
    models |= {k: (lambda mk=mk: _Wrap(mk())) for k, mk in _sk_models().items()}
    out, oofs = {}, {}
    for name, mk in models.items():
        out[name], oofs[name] = _lro(mk, X, y, groups)
    ok = ~np.isnan(oofs["np_logistic"])
    blend = np.mean([oofs[k][ok] for k in ("np_logistic", "sk_hist_gbm", "sk_extra_trees")], axis=0)
    lo, hi = bootstrap_auc_ci(y[ok], blend)
    out["blend_logistic+gbm+extratrees"] = {"auc": round(auc(y[ok], blend), 3), "ci95": [round(lo, 3), round(hi, 3)],
                                            "brier": round(brier(y[ok], blend), 4)}
    return {"task": task, "n": len(rows), "leave_region_out": dict(sorted(out.items(), key=lambda kv: -kv[1]["auc"]))}


def main() -> None:
    res = [run("flood", FLOOD_FILES), run("fire", FIRE_FILES)]
    Path("validation_data/benchmark.json").write_text(json.dumps(res, indent=2))
    for r in res:
        print(r["task"], r["n"])
        for k, v in r["leave_region_out"].items():
            print(f"  {k:32s} auc {v['auc']:.3f} {v['ci95']} brier {v['brier']}")


if __name__ == "__main__":
    main()
