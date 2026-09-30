"""Explainable risk scoring from spectral-index statistics.

Scores are 0-100 *screening* heuristics, not calibrated probabilities. Every score
carries the factors that produced it so a human (or an LLM) can explain it.
"""

from __future__ import annotations

from dataclasses import dataclass

from .indices import masked_mean, nbr, ndbi, ndvi, ndwi
from .scene import Scene


@dataclass
class Features:
    water_fraction: float      # share of valid pixels with NDWI > 0.1
    mean_ndvi: float
    mean_nbr: float
    dry_fuel_fraction: float   # vegetated-but-dry: NDVI in [0.2, 0.5) and NBR < 0.25 (NDVI < 0.2 is bare ground or built surface)
    built_fraction: float      # NDBI > 0.05 and NDVI < 0.25
    valid_fraction: float
    veg_fraction: float = 0.0   # NDVI > 0.3: any vegetation is potential fuel load
    veg_mean_nbr: float = float("nan")  # mean NBR over vegetated pixels only (fuel dryness)


@dataclass
class RiskScore:
    name: str
    score: float
    level: str
    factors: list[str]


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
            (((n_ndvi >= 0.2) & (n_ndvi < 0.5) & (n_nbr < 0.25)) & v).sum() / n_valid
        ),
        built_fraction=float((((n_ndbi > 0.05) & (n_ndvi < 0.25)) & v).sum() / n_valid),
        valid_fraction=scene.valid_fraction,
        veg_fraction=float(((n_ndvi > 0.3) & v).sum() / n_valid),
        veg_mean_nbr=masked_mean(n_nbr, (n_ndvi >= 0.2) & v),
    )
