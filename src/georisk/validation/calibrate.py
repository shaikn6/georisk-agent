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
from .ensemble import KNN, ZMAX, Ensemble, Forest, SkForest, dump_model
from .metrics import (
    Logistic,
    auc,
    bootstrap_auc_ci,
    brier,
    cross_validate,
    ece,
    leave_group_out,
    platt_apply,
    platt_fit,
    reliability_table,
)

TASKS = {
    "flood": {
        "hazard": FloodHazard,
        "features": ["water_fraction", "built_fraction", "mean_ndvi", "t_mean_slope_deg", "t_p90_slope_deg",
                     "t_relief_m", "t_center_percentile", "t_low_lying_fraction",
                     # height above the nearest stream (hydrology.py): +0.012 AUC, Ohio holdout +0.03..0.05
                     "h_hand_mean_m", "h_hand_min_m", "h_near_stream_fraction", "h_dist_to_stream_m", "h_log_stream_area",
                     "h_hand_big_mean_m"],
        "meaning": "P(site lies in a FEMA Special Flood Hazard Area)",
    },
    "solar": {
        "experimental": True,
        "hazard": None,  # an opportunity, not a hazard: no hand-written baseline, and no imagery (panels would leak the label)
        "kind": "opportunity",
        "features": ["t_mean_slope_deg", "t_p90_slope_deg", "t_relief_m", "t_center_percentile", "t_low_lying_fraction",
                     "h_hand_mean_m", "h_hand_min_m", "h_near_stream_fraction", "h_dist_to_stream_m", "h_log_stream_area", "h_hand_big_mean_m",
                     "o_elev_mean_m", "o_tpi_m", "o_roughness_m", "o_south_exposure", "o_flat_fraction",
                     # high-voltage grid proximity (grid.py): where utility-scale solar can actually connect
                     "g_dist_km", "g_dist_230_km", "g_n_lines_10km", "g_max_kv_10km"],
        "meaning": "Terrain and high-voltage-grid similarity to existing US utility-scale solar plants (not economics, irradiance, land price or permitting)",
    },
    "fire": {
        "hazard": WildfireHazard,
        "features": ["dry_fuel_fraction", "veg_fraction", "veg_mean_nbr", "built_fraction", "mean_ndvi",
                     "mean_nbr", "t_mean_slope_deg", "t_p90_slope_deg", "t_relief_m",
                     # elevation, ridge/valley position, roughness and aspect (hydrology.topo_features): calibration error 0.108 -> 0.043
                     "o_elev_mean_m", "o_tpi_m", "o_roughness_m", "o_south_exposure"],
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
    extra: dict[tuple[float, float], dict] = {}
    for fname, prefix in (("hydro.jsonl", "h_"), ("topo.jsonl", "o_"), ("grid.jsonl", "g_")):  # separately collected DEM features, joined by coordinates
        fp = Path("validation_data") / fname
        if not fp.exists():
            continue
        for line in fp.read_text().splitlines():
            h = json.loads(line) if line.strip() else None
            if h and not any(k.endswith("_error") for k in h):
                extra.setdefault((h["lat"], h["lon"]), {}).update({k: v for k, v in h.items() if k.startswith(prefix)})
    for r in rows:
        r.update(extra.get((r["lat"], r["lon"]), {}))
    keep = [r for r in rows if "terrain_error" not in r and r.get("valid_fraction", 1.0) >= 0.5 and "t_relief_m" in r]
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


def _nested_platt(oof: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Recalibrated held-out predictions: each region is corrected by a curve fitted on the OTHER regions only."""
    fixed = np.full(len(y), np.nan)
    ok = ~np.isnan(oof)
    for g in np.unique(groups[ok]):
        te, tr = ok & (groups == g), ok & (groups != g)
        if tr.sum() > 30 and len(np.unique(y[tr])) == 2:
            fixed[te] = platt_apply(oof[te], *platt_fit(oof[tr], y[tr]))
    return fixed


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
    heur = np.array([spec["hazard"]().score(HazardContext(*_feat_obj(r))).score for r in rows]) if spec["hazard"] else None
    h_auc, (h_lo, h_hi) = (auc(y, heur), bootstrap_auc_ci(y, heur)) if heur is not None else (None, (None, None))
    cv = cross_validate(X, y, k=5, l2=L2)  # optimistic: neighbours leak between folds (reported, not gated)
    lg = leave_group_out(X, y, [r["region"] for r in rows], l2=L2)  # the honest test: unseen regions
    groups = np.array([r["region"] for r in rows])
    members = {"logistic": lambda: Logistic(L2, zmax=ZMAX), "forest": Forest, "knn": KNN, "ensemble": lambda: Ensemble(L2)}
    if _HAS_SKLEARN:  # research extra: trees are trained with scikit-learn, then served as plain numpy
        members |= {"sk_forest": SkForest, "blend": lambda: Ensemble(L2, ("logistic", "sk_forest"))}
    lro = {name: _lro(mk, X, y, groups) for name, mk in members.items()}
    ohio = np.array([r["region"] in OHIO or r["region"] == "OH" for r in rows])  # city regions, or the state code (solar)
    ohio_case = None
    if ohio.sum() >= 15 and len(np.unique(y[ohio])) == 2 and len(np.unique(y[~ohio])) == 2:
        ohio_case = {"n_ohio": int(ohio.sum()), "trained_on": "all non-Ohio regions", "by_model": {}}
        for name, mk in members.items():
            pr = mk().fit(X[~ohio], y[~ohio]).predict_proba(X[ohio])
            lo, hi = bootstrap_auc_ci(y[ohio], pr)
            ohio_case["by_model"][name] = {"auc": round(auc(y[ohio], pr), 3), "auc_ci95": [round(lo, 3), round(hi, 3)],
                                           "brier": round(brier(y[ohio], pr), 4),
                                           "brier_baseline": round(brier(y[ohio], np.full(ohio.sum(), y[~ohio].mean())), 4)}
        ohio_case["heuristic_auc"] = round(auc(y[ohio], heur[ohio]), 3) if heur is not None else None
    # Serving rule: the product is a probability, so among candidates within 0.01 AUC of the best (ranking is a tie),
    # serve the one with the best Brier score after the recalibration gate, i.e. what users actually receive.
    best_auc = max(lro[n][1]["auc"] for n in members)
    final_brier = {}
    for n in members:
        o = lro[n][0]
        k = ~np.isnan(o)
        raw, fx = brier(y[k], o[k]), _nested_platt(o, y, groups)
        final_brier[n] = min(raw, brier(y[~np.isnan(fx)], fx[~np.isnan(fx)])) if (~np.isnan(fx)).any() else raw
    eligible = [n for n in members if lro[n][1]["auc"] >= best_auc - 0.01]
    chosen = min(eligible, key=lambda n: final_brier[n])
    ens = lro[chosen][1]
    oof_chosen = lro[chosen][0]
    ok_o = ~np.isnan(oof_chosen)
    fixed = _nested_platt(oof_chosen, y, groups)
    ok_f = ~np.isnan(fixed)
    recal = {"ece_before": round(ece(y[ok_f], oof_chosen[ok_f]), 4), "ece_after": round(ece(y[ok_f], fixed[ok_f]), 4),
             "brier_before": round(brier(y[ok_f], oof_chosen[ok_f]), 4), "brier_after": round(brier(y[ok_f], fixed[ok_f]), 4),
             "auc_after": round(auc(y[ok_f], fixed[ok_f]), 3)}
    a_cal, b_cal = platt_fit(oof_chosen[ok_o], y[ok_o])
    lg = {**lg, "auc": ens["auc"], "auc_ci95": ens["auc_ci95"], "brier": ens["brier"], "brier_baseline": ens["brier_baseline"]}
    passed = lg["auc"] >= GATE["min_auc"] and lg["auc_ci95"][0] > GATE["min_ci_low"] and lg["brier"] < lg["brier_baseline"]
    rep = {
        "task": task, "n": len(rows), "positives": int(y.sum()), "meaning": spec["meaning"],
        "heuristic": {"auc": round(h_auc, 3), "auc_ci95": [round(h_lo, 3), round(h_hi, 3)]} if heur is not None else None,
        "model_random_kfold": {"auc": round(cv["auc"], 3), "auc_ci95": [round(v, 3) for v in cv["auc_ci95"]],
                               "brier": round(cv["brier"], 4), "brier_baseline": round(cv["brier_baseline"], 4)},
        "leave_region_out_by_model": {n: v[1] for n, v in lro.items()},
        "model_leave_region_out": {**ens, "per_region_auc": {k: round(v, 2) for k, v in lg["per_group_auc"].items()}},
        "served_model": chosen, "selection": {"best_auc": best_auc, "eligible": eligible, "final_brier": {n: round(v, 4) for n, v in final_brier.items()}}, "recalibration": recal, "ohio_holdout": ohio_case, "gate_applied_to": "model_leave_region_out (served model)", "gate": GATE, "shipped": bool(passed),
    }
    if passed:
        m = members[chosen]().fit(X, y)
        helps = recal["ece_after"] < recal["ece_before"] and recal["brier_after"] <= recal["brier_before"]  # ship only a clear win
        rep["recalibration"]["shipped"] = bool(helps)
        rep["artifact"] = {"features": spec["features"], "fill": fill, "model": dump_model(chosen, m), "meaning": spec["meaning"],
                           "domain": _domain(rows), "kind": spec.get("kind", "hazard")}
        if spec.get("experimental"):  # the weaknesses travel with the model, so no consumer can drop them
            oh = (ohio_case or {}).get("by_model", {}).get(chosen)
            rep["artifact"]["caveat"] = (
                f"Experimental: held-out AUC {ens['auc']:.2f} across states"
                + (f", but {oh['auc']:.2f} in Ohio (only {ohio_case['n_ohio']} test sites)" if oh else "")
                + ". In flat farmland, terrain and grid distance barely separate solar sites from neighbouring fields, so treat it as a rough screen.")
        if helps:
            rep["artifact"]["calibrator"] = {"a": round(a_cal, 4), "b": round(b_cal, 4)}
        shown = fixed[ok_f] if helps else oof_chosen[ok_f]  # the probabilities users will actually see, on held-out regions
        rep["artifact"]["reliability"] = reliability_table(shown, y[ok_f])
    return rep


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--flood", type=Path, nargs="+", default=[Path("validation_data/flood.jsonl"), Path("validation_data/flood2.jsonl"), Path("validation_data/flood3.jsonl"), Path("validation_data/flood_ohio.jsonl")])
    p.add_argument("--fire", type=Path, nargs="+", default=[Path("validation_data/fire.jsonl"), Path("validation_data/fire2.jsonl")])
    p.add_argument("--solar", type=Path, nargs="+", default=[Path("validation_data/solar.jsonl")])
    p.add_argument("--report", type=Path, default=Path("validation_data/report.json"))
    p.add_argument("--artifact", type=Path, default=Path("src/georisk/data/calibration.json"))
    a = p.parse_args()
    reports, artifact = [], {}
    for task, paths in (("flood", a.flood), ("fire", a.fire), ("solar", a.solar)):
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
