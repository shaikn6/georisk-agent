"""Deterministic synthetic scenes so the pipeline runs offline and in CI.

Each archetype paints a coarse land-cover layout and maps classes to typical reflectance.
This is a *test fixture*, not a claim about real imagery.
"""

from __future__ import annotations

import numpy as np

from .scene import Scene

# Typical surface reflectance (green, red, nir, swir1, swir2) per class.
_CLASS_REFLECTANCE = {
    "water": (0.06, 0.04, 0.02, 0.01, 0.01),
    "forest": (0.05, 0.03, 0.42, 0.16, 0.07),
    "dry_grass": (0.10, 0.14, 0.24, 0.28, 0.22),
    "urban": (0.14, 0.16, 0.20, 0.26, 0.24),
}
_ORDER = ["water", "forest", "dry_grass", "urban"]

ARCHETYPES: dict[str, tuple[float, float, float, float]] = {
    # water, forest, dry_grass, urban fractions
    "floodplain": (0.45, 0.10, 0.25, 0.20),
    "wildland_urban": (0.02, 0.18, 0.55, 0.25),
    "dense_urban": (0.02, 0.05, 0.08, 0.85),
    "healthy_forest": (0.03, 0.90, 0.05, 0.02),
}


def make_scene(archetype: str, size: int = 96, seed: int = 7) -> Scene:
    if archetype not in ARCHETYPES:
        raise ValueError(f"unknown archetype {archetype!r}; choose from {sorted(ARCHETYPES)}")
    rng = np.random.default_rng(seed)
    fractions = np.array(ARCHETYPES[archetype])
    # Smooth-ish class map: coarse random field, thresholded by cumulative fractions.
    coarse = rng.random((size // 8, size // 8))
    field = np.kron(coarse, np.ones((8, 8)))
    ranks = field.argsort(axis=None).argsort(axis=None).reshape(field.shape) / field.size
    edges = np.cumsum(fractions)
    labels = np.searchsorted(edges, ranks, side="right").clip(0, len(_ORDER) - 1)
    bands = {}
    for i, name in enumerate(["green", "red", "nir", "swir1", "swir2"]):
        base = np.zeros((size, size))
        for k, cls in enumerate(_ORDER):
            base[labels == k] = _CLASS_REFLECTANCE[cls][i]
        bands[name] = np.clip(base + rng.normal(0, 0.008, base.shape), 0.0, 1.0)
    valid = np.ones((size, size), dtype=bool)
    return Scene(
        bands=bands,
        valid=valid,
        source=f"synthetic:{archetype}",
        acquired=None,
        cloud_cover=0.0,
        meta={"seed": seed, "size": size},
    )
