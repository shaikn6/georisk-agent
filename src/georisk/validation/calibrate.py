"""Evaluate the heuristic scores against ground truth and fit calibrated models.

A model is only *shipped* if it clears a strict gate on out-of-fold data. Otherwise the report says
so and the service keeps serving the heuristic score without a probability.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..hazards import HazardContext
from ..hazards.flood import FloodHazard
from ..hazards.wildfire import WildfireHazard
from ..scoring import Features
from ..terrain import TerrainFeatures
from .dataset import OHIO
from .ensemble import KNN, Ensemble, Forest, SkForest, dump_model
from .metrics import Logistic, auc, bootstrap_auc_ci, brier, cross_validate, leave_group_out

TASKS = {
    "flood": {
        "hazard": FloodHazard,
        "features": ["water_fraction", "built_fraction", "mean_ndvi", "t_mean_slope_deg", "t_p90_slope_deg",
                     "t_relief_m", "t_center_percentile", "t_low_lying_fraction"],
        "meaning": "P(site lies in a FEMA Special Flood Hazard Area)",
    },
    "fire": {
        "hazard": WildfireHazard,
        "features": ["dry_fuel_fraction", "veg_fraction", "veg_mean_nbr", "built_fraction", "mean_ndvi",
                     "mean_nbr", "t_mean_slope_deg", "t_p90_slope_deg", "t_relief_m"],
        "meaning": "P(site lies inside a >=100-acre fire perimeter from 2005-2018)",
    },
}
try:
    import sklearn  # noqa: F401

    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False
L2 = 3.0  # fixed a priori: no hyper-parameter search on a small sample
GATE = {"min_auc": 0.6, "min_ci_low": 0.5}


def _feat_obj(row: dict) -> tuple[Features, TerrainFeatures]:
    f = Features(**{k: row[k] for k in Features.__dataclass_fields__ if k in row})
    t = TerrainFeatures(row["t_mean_slope_deg"], row["t_p90_slope_deg"], row["t_relief_m"],
                        row["t_center_percentile"], row["t_low_lying_fraction"])
    return f, t


def load_rows(*paths: Path) -> list[dict]:
    seen, rows = set(), []
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            r = json.loads(line) if line.strip() else None
            if r and (r["lat"], r["lon"]) not in seen:  # the same point sampled twice must not leak across folds
                seen.add((r["lat"], r["lon"]))
                rows.append(r)
    keep = [r for r in rows if "terrain_error" not in r and r.get("valid_fraction", 0) >= 0.5 and "t_relief_m" in r]
    return keep


def build_matrix(rows: list[dict], cols: list[str]) -> tuple[np.ndarray, np.ndarray, dict]:
    X = np.array([[r.get(c, np.nan) if r.get(c) is not None else np.nan for c in cols] for r in rows], dtype=float)
    fill = {c: float(np.nanmedian(X[:, i])) if np.isfinite(X[:, i]).any() else 0.0 for i, c in enumerate(cols)}
    for i, c in enumerate(cols):
        X[np.isnan(X[:, i]), i] = fill[c]
    return X, np.array([r["label"] for r in rows]), fill


def _domain(rows: list[dict], pad: float = 1.0) -> list[float]:
    """[lat0, lat1, lon0, lon1] of the training points (+pad deg): outside it the model is extrapolating."""
    lats, lons = [r["lat"] for r in rows], [r["lon"] for r in rows]
    return [round(min(lats) - pad, 2), round(max(lats) + pad, 2), round(min(lons) - pad, 2), round(max(lons) + pad, 2)]


def _lro(make, X, y, groups) -> tuple[np.ndarray, dict]:
    """Leave-region-out out-of-fold predictions for any model factory, with the same metrics."""
    oof = np.full(len(y), np.nan)
    for g in np.unique(groups):
        te = groups == g
        if len(np.unique(y[~te])) == 2:
            oof[te] = make().fit(X[~te], y[~te]).predict_proba(X[te])
    ok = ~np.isnan(oof)
    lo, hi = bootstrap_auc_ci(y[ok], oof[ok])
    return oof, {"auc": round(auc(y[ok], oof[ok]), 3), "auc_ci95": [round(lo, 3), round(hi, 3)],
                 "brier": round(brier(y[ok], oof[ok]), 4), "brier_baseline": round(brier(y[ok], np.full(ok.sum(), y[ok].mean())), 4)}


def evaluate(task: str, *paths: Path) -> dict:
    spec = TASKS[task]
    rows = load_rows(*paths)
    if len(rows) < 30:
        return {"task": task, "n": len(rows), "shipped": False, "reason": "too few usable samples"}
    X, y, fill = build_matrix(rows, spec["features"])
    heur = np.array([spec["hazard"]().score(HazardContext(*_feat_obj(r))).score for r in rows])
    h_auc, (h_lo, h_hi) = auc(y, heur), bootstrap_auc_ci(y, heur)
    cv = cross_validate(X, y, k=5, l2=L2)  # optimistic: neighbours leak between folds (reported, not gated)
    lg = leave_group_out(X, y, [r["region"] for r in rows], l2=L2)  # the honest test: unseen regions
    groups = np.array([r["region"] for r in rows])
    members = {"logistic": lambda: Logistic(L2), "forest": Forest, "knn": KNN, "ensemble": lambda: Ensemble(L2)}
    if _HAS_SKLEARN:  # research extra: trees are trained with scikit-learn, then served as plain numpy
        members |= {"sk_forest": SkForest, "blend": lambda: Ensemble(L2, ("logistic", "sk_forest"))}
    lro = {name: _lro(mk, X, y, groups) for name, mk in members.items()}
    ohio = np.array([r["region"] in OHIO for r in rows])
    ohio_case = None
    if ohio.sum() >= 15 and len(np.unique(y[ohio])) == 2 and len(np.unique(y[~ohio])) == 2:
        ohio_case = {"n_ohio": int(ohio.sum()), "trained_on": "all non-Ohio regions", "by_model": {}}
        for name, mk in members.items():
            pr = mk().fit(X[~ohio], y[~ohio]).predict_proba(X[ohio])
            lo, hi = bootstrap_auc_ci(y[ohio], pr)
            ohio_case["by_model"][name] = {"auc": round(auc(y[ohio], pr), 3), "auc_ci95": [round(lo, 3), round(hi, 3)],
                                           "brier": round(brier(y[ohio], pr), 4),
                                           "brier_baseline": round(brier(y[ohio], np.full(ohio.sum(), y[~ohio].mean())), 4)}
        hh = heur[ohio]
        ohio_case["heuristic_auc"] = round(auc(y[ohio], hh), 3)
    # The served model is whichever candidate has the best leave-region-out AUC (all four are reported).
    # With ~100 samples they are usually statistically tied; the ensemble wins only when it earns it.
    chosen = max(members, key=lambda n: lro[n][1]["auc"])
    ens = lro[chosen][1]
    lg = {**lg, "auc": ens["auc"], "auc_ci95": ens["auc_ci95"], "brier": ens["brier"], "brier_baseline": ens["brier_baseline"]}
    passed = lg["auc"] >= GATE["min_auc"] and lg["auc_ci95"][0] > GATE["min_ci_low"] and lg["brier"] < lg["brier_baseline"]
    rep = {
        "task": task, "n": len(rows), "positives": int(y.sum()), "meaning": spec["meaning"],
        "heuristic": {"auc": round(h_auc, 3), "auc_ci95": [round(h_lo, 3), round(h_hi, 3)]},
        "model_random_kfold": {"auc": round(cv["auc"], 3), "auc_ci95": [round(v, 3) for v in cv["auc_ci95"]],
                               "brier": round(cv["brier"], 4), "brier_baseline": round(cv["brier_baseline"], 4)},
        "leave_region_out_by_model": {n: v[1] for n, v in lro.items()},
        "model_leave_region_out": {**ens, "per_region_auc": {k: round(v, 2) for k, v in lg["per_group_auc"].items()}},
        "served_model": chosen, "ohio_holdout": ohio_case, "gate_applied_to": "model_leave_region_out (served model)", "gate": GATE, "shipped": bool(passed),
    }
    if passed:
        m = members[chosen]().fit(X, y)
        rep["artifact"] = {"features": spec["features"], "fill": fill, "model": dump_model(chosen, m), "meaning": spec["meaning"],
                           "domain": _domain(rows)}
    return rep


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--flood", type=Path, nargs="+", default=[Path("validation_data/flood.jsonl"), Path("validation_data/flood2.jsonl"), Path("validation_data/flood_ohio.jsonl")])
    p.add_argument("--fire", type=Path, default=Path("validation_data/fire.jsonl"))
    p.add_argument("--report", type=Path, default=Path("validation_data/report.json"))
    p.add_argument("--artifact", type=Path, default=Path("src/georisk/data/calibration.json"))
    a = p.parse_args()
    reports, artifact = [], {}
    for task, paths in (("flood", a.flood), ("fire", [a.fire])):
        if not any(p.exists() for p in paths):
            continue
        r = evaluate(task, *paths)
        if "artifact" in r:
            artifact[task] = r.pop("artifact")
        reports.append(r)
        print(json.dumps(r, indent=2))
    a.report.write_text(json.dumps(reports, indent=2))
    if artifact:
        a.artifact.parent.mkdir(parents=True, exist_ok=True)
        a.artifact.write_text(json.dumps(artifact))
        print(f"wrote {a.artifact} for: {', '.join(artifact)}")
    else:
        a.artifact.unlink(missing_ok=True)
        print("no model cleared the gate: nothing shipped")


if __name__ == "__main__":
    main()
