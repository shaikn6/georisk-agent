"""Explainable risk scoring from spectral-index statistics.

Scores are 0-100 *screening* heuristics, not calibrated probabilities. Every score
carries the factors that produced it so a human (or an LLM) can explain it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .indices import masked_mean, nbr, ndbi, ndvi, ndwi
from .scene import Scene


@dataclass
class Features:
    water_fraction: float      # share of valid pixels with NDWI > 0.1
    mean_ndvi: float
    mean_nbr: float
    dry_fuel_fraction: float   # vegetated-but-dry: NDVI in (0.1, 0.5) and NBR < 0.25
    built_fraction: float      # NDBI > 0.05 and NDVI < 0.25
    valid_fraction: float


@dataclass
class RiskScore:
    name: str
    score: float
    level: str
    factors: list[str]


def _clip100(x: float) -> float:
    return float(np.clip(x, 0.0, 100.0))


def _level(score: float) -> str:
    return "low" if score < 25 else "moderate" if score < 50 else "high" if score < 75 else "severe"


def extract_features(scene: Scene) -> Features:
    b = scene.bands
    v = scene.valid
    n_ndvi = ndvi(b["nir"], b["red"])
    n_ndwi = ndwi(b["green"], b["nir"])
    n_ndbi = ndbi(b["swir1"], b["nir"])
    n_nbr = nbr(b["nir"], b["swir2"])
    n_valid = max(int(v.sum()), 1)
    return Features(
        water_fraction=float(((n_ndwi > 0.1) & v).sum() / n_valid),
        mean_ndvi=masked_mean(n_ndvi, v),
        mean_nbr=masked_mean(n_nbr, v),
        dry_fuel_fraction=float(
            (((n_ndvi > 0.1) & (n_ndvi < 0.5) & (n_nbr < 0.25)) & v).sum() / n_valid
        ),
        built_fraction=float((((n_ndbi > 0.05) & (n_ndvi < 0.25)) & v).sum() / n_valid),
        valid_fraction=scene.valid_fraction,
    )


def flood_risk(f: Features) -> RiskScore:
    # Standing water in the footprint plus a large impervious share (poor drainage).
    score = _clip100(140 * f.water_fraction + 35 * f.built_fraction)
    factors = [f"{f.water_fraction:.0%} of the area reads as open water (NDWI)"]
    if f.built_fraction > 0.3:
        factors.append(f"{f.built_fraction:.0%} impervious surface limits drainage")
    return RiskScore("flood", score, _level(score), factors)


def wildfire_risk(f: Features) -> RiskScore:
    # Dry fuel dominates; adjacent built area raises exposure of assets.
    dryness = 0.0 if np.isnan(f.mean_nbr) else max(0.0, (0.3 - f.mean_nbr) / 0.5)
    score = _clip100(110 * f.dry_fuel_fraction + 30 * dryness + 20 * f.built_fraction * f.dry_fuel_fraction)
    factors = [f"{f.dry_fuel_fraction:.0%} of the area is vegetated but dry (NDVI/NBR)"]
    if not np.isnan(f.mean_nbr):
        factors.append(f"mean NBR {f.mean_nbr:.2f} (lower = drier fuel)")
    return RiskScore("wildfire", score, _level(score), factors)


def urban_encroachment(f: Features) -> RiskScore:
    score = _clip100(115 * f.built_fraction)
    factors = [f"{f.built_fraction:.0%} of the area is built-up (NDBI/NDVI)"]
    return RiskScore("urban_density", score, _level(score), factors)


def score_all(f: Features) -> list[RiskScore]:
    return [flood_risk(f), wildfire_risk(f), urban_encroachment(f)]


def features_dict(f: Features) -> dict:
    return asdict(f)
