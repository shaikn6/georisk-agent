"""Optional calibrated probabilities from models fitted on public ground truth.

Every failure mode returns an empty result: a missing, corrupt or inapplicable model must never break an
assessment, it just means no probability is reported.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from functools import lru_cache
from importlib import resources

import numpy as np

from .scoring import Features
from .terrain import TerrainFeatures
from .validation.ensemble import load_model
from .validation.metrics import platt_apply

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_artifact() -> dict:
    try:
        raw = resources.files("georisk").joinpath("data/calibration.json").read_text()
        return json.loads(raw)
    except FileNotFoundError:
        return {}
    except Exception as exc:  # noqa: BLE001 - corrupt file: serve without probabilities, but say so
        log.warning("calibration artifact unreadable, ignoring: %s", exc)
        return {}


def _vector(spec: dict, features: Features, terrain: TerrainFeatures, hydro=None) -> np.ndarray:
    row = {**asdict(features), **{f"t_{k}": v for k, v in asdict(terrain).items()}}
    if hydro is not None:
        row |= {f"h_{k}": v for k, v in asdict(hydro).items()}  # absent hydrology falls back to the training median via `fill`
    out = []
    for c in spec["features"]:
        v = row.get(c)
        out.append(spec["fill"][c] if v is None or (isinstance(v, float) and np.isnan(v)) else float(v))
    return np.array([out])


LABELS = {
    "water_fraction": ("Open water", "%"), "built_fraction": ("Built-up surface", "%"), "mean_ndvi": ("Vegetation greenness (NDVI)", ""),
    "mean_nbr": ("Burn/dryness index (NBR)", ""), "dry_fuel_fraction": ("Dry vegetation", "%"), "veg_fraction": ("Dense vegetation", "%"),
    "veg_mean_nbr": ("Vegetation dryness (NBR)", ""), "t_mean_slope_deg": ("Mean slope", "deg"), "t_p90_slope_deg": ("Steep-slope extent", "deg"),
    "t_relief_m": ("Local relief", "m"), "t_center_percentile": ("Elevation vs surroundings", "%"), "t_low_lying_fraction": ("Low-lying ground", "%"),
    "h_hand_mean_m": ("Height above nearest stream", "m"), "h_hand_min_m": ("Lowest point above stream", "m"),
    "h_near_stream_fraction": ("Ground within 3 m of a stream", "%"), "h_dist_to_stream_m": ("Distance to nearest stream", "m"),
    "h_log_stream_area": ("Size of the stream it drains to (log km2)", ""), "h_hand_big_mean_m": ("Height above large rivers", "m"),
}
_PERCENT = {"water_fraction", "built_fraction", "dry_fuel_fraction", "veg_fraction", "t_center_percentile", "t_low_lying_fraction", "h_near_stream_fraction"}


# Correlated features are occluded together: one at a time they produce offsetting, misleading effects.
GROUPS = {
    "Terrain steepness": ("t_mean_slope_deg", "t_p90_slope_deg"),
    "Low-lying position": ("t_center_percentile", "t_low_lying_fraction"),
    "Height above streams": ("h_hand_mean_m", "h_hand_min_m", "h_hand_big_mean_m", "h_near_stream_fraction"),
    "Stream proximity and size": ("h_dist_to_stream_m", "h_log_stream_area"),
    "Fuel dryness": ("dry_fuel_fraction", "veg_mean_nbr", "mean_nbr"),
}


def _units(spec: dict) -> list[tuple[str, tuple[str, ...]]]:
    grouped = {c for g in GROUPS.values() if all(m in spec["features"] for m in g) for c in g}
    units = [(label, g) for label, g in GROUPS.items() if all(m in spec["features"] for m in g)]
    return units + [(LABELS.get(c, (c, ""))[0], (c,)) for c in spec["features"] if c not in grouped]


class Recalibrated:
    """A model plus its recalibration curve; the probability shown and the drivers explained are the same number."""

    def __init__(self, model, calibrator: dict | None) -> None:
        self.model, self.cal = model, calibrator

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = self.model.predict_proba(X)
        return platt_apply(p, self.cal["a"], self.cal["b"]) if self.cal else p


def similar_sites(spec: dict, p: float) -> dict | None:
    """What happened to held-out sites in testing whose predicted probability was close to this one."""
    for b in spec.get("reliability", []):
        if b["lo"] <= p < b["hi"] or (p >= 1.0 and b["hi"] >= 1.0):
            return {"n": b["n"], "observed_pct": round(b["rate"] * 100), "ci95_pct": [round(b["ci"][0] * 100), round(b["ci"][1] * 100)],
                    "band_pct": [round(b["lo"] * 100), round(b["hi"] * 100)]}
    return None


def explain(model, spec: dict, x: np.ndarray, top: int = 5) -> list[dict]:
    """What-if drivers: how much does the probability change if a feature (group) were a typical training site's value?

    Positive effect = pushes the probability up relative to a typical site. Occlusion ignores interactions and
    says nothing about causation; it explains the model, not the world.
    """
    base = float(model.predict_proba(x)[0])
    idx = {c: i for i, c in enumerate(spec["features"])}
    rows = []
    for label, members in _units(spec):
        alt = x.copy()
        parts = []
        for c in members:
            alt[0, idx[c]] = spec["fill"][c]
            name, unit = LABELS.get(c, (c, ""))
            k = 100.0 if c in _PERCENT else 1.0
            parts.append({"name": name, "unit": unit, "value": round(float(x[0, idx[c]]) * k, 2),
                          "typical": round(float(spec["fill"][c]) * k, 2)})
        rows.append({"label": label, "effect_pp": round((base - float(model.predict_proba(alt)[0])) * 100, 1), "parts": parts})
    rows.sort(key=lambda r: -abs(r["effect_pp"]))
    return [r for r in rows[:top] if abs(r["effect_pp"]) >= 0.5]


def outside_domain(lat: float, lon: float, artifact: dict | None = None) -> list[str]:
    """Tasks that have a fitted model whose training region does not contain (lat, lon)."""
    art = load_artifact() if artifact is None else artifact
    return [t for t, spec in art.items()
            if (d := spec.get("domain")) and not (d[0] <= lat <= d[1] and d[2] <= lon <= d[3])]


def calibrated_probabilities(features: Features, terrain: TerrainFeatures | None, artifact: dict | None = None,
                             lat: float | None = None, lon: float | None = None, hydro=None) -> dict:
    """{task: {"probability": p, "meaning": str}} for every task with a shipped model. Needs terrain; skips
    tasks whose training domain does not contain (lat, lon)."""
    art = load_artifact() if artifact is None else artifact
    if terrain is None or not art:
        return {}
    out = {}
    for task, spec in art.items():
        d = spec.get("domain")
        if d and lat is not None and lon is not None and not (d[0] <= lat <= d[1] and d[2] <= lon <= d[3]):
            continue  # outside the validated region: no probability beats a made-up one
        try:
            model = Recalibrated(load_model(spec["model"]), spec.get("calibrator"))  # undoes the squeeze toward 0.5
            x = _vector(spec, features, terrain, hydro)
            p = float(model.predict_proba(x)[0])
            out[task] = {"probability": round(p, 3), "meaning": spec["meaning"], "drivers": explain(model, spec, x),
                         "evidence": similar_sites(spec, p)}
        except Exception as exc:  # noqa: BLE001
            log.warning("calibrated %s probability failed: %s", task, exc)
    return out
