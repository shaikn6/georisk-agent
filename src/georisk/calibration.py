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


def _vector(spec: dict, features: Features, terrain: TerrainFeatures) -> np.ndarray:
    row = {**asdict(features), **{f"t_{k}": v for k, v in asdict(terrain).items()}}
    out = []
    for c in spec["features"]:
        v = row.get(c)
        out.append(spec["fill"][c] if v is None or (isinstance(v, float) and np.isnan(v)) else float(v))
    return np.array([out])


def outside_domain(lat: float, lon: float, artifact: dict | None = None) -> list[str]:
    """Tasks that have a fitted model whose training region does not contain (lat, lon)."""
    art = load_artifact() if artifact is None else artifact
    return [t for t, spec in art.items()
            if (d := spec.get("domain")) and not (d[0] <= lat <= d[1] and d[2] <= lon <= d[3])]


def calibrated_probabilities(features: Features, terrain: TerrainFeatures | None, artifact: dict | None = None,
                             lat: float | None = None, lon: float | None = None) -> dict:
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
            p = float(load_model(spec["model"]).predict_proba(_vector(spec, features, terrain))[0])
            out[task] = {"probability": round(p, 3), "meaning": spec["meaning"]}
        except Exception as exc:  # noqa: BLE001
            log.warning("calibrated %s probability failed: %s", task, exc)
    return out
