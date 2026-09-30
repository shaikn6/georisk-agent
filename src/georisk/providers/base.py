"""Provider contract. A provider turns (bbox, params) into a Scene or raises ProviderError."""

from __future__ import annotations

from typing import Protocol

from ..aoi import BBox
from ..scene import Scene


class ProviderError(RuntimeError):
    """Any failure to obtain imagery from one provider (network, auth, parsing)."""


class NoSceneFound(ProviderError):
    """The provider works but has no acceptable scene. Not worth retrying."""


class SceneProvider(Protocol):
    name: str

    def load(self, bbox: BBox, *, days: int, max_cloud: float, grid: int) -> Scene: ...

    def probe(self) -> None:
        """Cheap reachability check; raise ProviderError if unhealthy."""
