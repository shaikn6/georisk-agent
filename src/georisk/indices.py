"""Spectral indices on Sentinel-2 surface reflectance (scaled 0-1)."""

from __future__ import annotations

import numpy as np

_EPS = 1e-9


def _norm_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype("float64")
    b = b.astype("float64")
    return (a - b) / (a + b + _EPS)


def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Vegetation greenness. High = dense healthy vegetation."""
    return _norm_diff(nir, red)


def ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """McFeeters water index. > 0 typically open water."""
    return _norm_diff(green, nir)


def ndbi(swir1: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Built-up index. High = impervious / bare built surfaces."""
    return _norm_diff(swir1, nir)


def nbr(nir: np.ndarray, swir2: np.ndarray) -> np.ndarray:
    """Normalised burn ratio. Low values = dry / burnt / low-moisture fuel."""
    return _norm_diff(nir, swir2)


def masked_mean(values: np.ndarray, valid: np.ndarray) -> float:
    """Mean over valid pixels; NaN if none are valid."""
    v = values[valid]
    return float(v.mean()) if v.size else float("nan")
