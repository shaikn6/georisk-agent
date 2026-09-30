"""Microsoft Planetary Computer (independent second source; token signing is keyless)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import planetary_computer
from pystac_client import Client

from ..aoi import BBox
from ..scene import Scene
from ._common import build_scene
from .base import NoSceneFound, ProviderError

URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
_ASSETS = {"green": "B03", "red": "B04", "nir": "B08", "swir1": "B11", "swir2": "B12"}


class PlanetaryComputerProvider:
    name = "planetary-computer"

    def _client(self) -> Client:
        return Client.open(URL, modifier=planetary_computer.sign_inplace, timeout=20)

    def probe(self) -> None:
        try:
            Client.open(URL, timeout=10)
        except Exception as exc:
            raise ProviderError(f"{self.name} unreachable: {exc}") from exc

    def load(self, bbox: BBox, *, days: int, max_cloud: float, grid: int) -> Scene:
        try:
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=days)
            items = list(
                self._client()
                .search(
                    collections=["sentinel-2-l2a"],
                    bbox=bbox.as_list(),
                    datetime=f"{start:%Y-%m-%d}/{end:%Y-%m-%d}",
                    query={"eo:cloud_cover": {"lt": max_cloud}},
                    max_items=25,
                )
                .items()
            )
            if not items:
                raise NoSceneFound(f"{self.name}: no scene under {max_cloud}% cloud in {days} days")
            best = min(items, key=lambda i: i.properties.get("eo:cloud_cover", 100.0))
            return build_scene(best, _ASSETS, "SCL", bbox, grid, self.name)
        except NoSceneFound:
            raise
        except Exception as exc:
            raise ProviderError(f"{self.name}: {exc}") from exc
