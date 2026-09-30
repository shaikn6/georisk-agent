"""Synthetic provider: only used when explicitly enabled; results are always flagged synthetic."""

from __future__ import annotations

from ..aoi import BBox
from ..scene import Scene
from ..synthetic import make_scene


class SyntheticProvider:
    name = "synthetic"

    def __init__(self, archetype: str = "wildland_urban") -> None:
        self.archetype = archetype

    def probe(self) -> None:
        return None

    def load(self, bbox: BBox, *, days: int, max_cloud: float, grid: int) -> Scene:
        return make_scene(self.archetype, size=max(8, (grid // 8) * 8))
