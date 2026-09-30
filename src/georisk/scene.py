"""Scene container shared by the live STAC loader and the synthetic generator."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

BANDS = ("green", "red", "nir", "swir1", "swir2")


@dataclass
class Scene:
    """Co-registered reflectance bands (0-1) plus a validity mask (True = usable pixel)."""

    bands: dict[str, np.ndarray]
    valid: np.ndarray
    source: str
    acquired: str | None = None
    cloud_cover: float | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        missing = [b for b in BANDS if b not in self.bands]
        if missing:
            raise ValueError(f"scene missing bands: {missing}")
        shapes = {a.shape for a in self.bands.values()} | {self.valid.shape}
        if len(shapes) != 1:
            raise ValueError(f"band/mask shape mismatch: {shapes}")

    @property
    def valid_fraction(self) -> float:
        return float(self.valid.mean())
