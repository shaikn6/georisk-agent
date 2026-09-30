"""Shared STAC-provider behaviour: search, pick, build. Concrete providers only supply config."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import ClassVar

from pystac_client import Client

from ..aoi import BBox
from ..scene import Scene
from ._common import build_scene, build_scenes
from .base import NoSceneFound, ProviderError


class StacProvider:
    """Subclasses set: name, url, assets (band->asset key), scl_asset, and may override _client()."""

    name = "stac"
    url = ""
    assets: ClassVar[dict[str, str]] = {}
    scl_asset = "scl"

    def _client(self) -> Client:
        return Client.open(self.url, timeout=20)

    def probe(self) -> None:
        try:
            Client.open(self.url, timeout=10)
        except Exception as exc:
            raise ProviderError(f"{self.name} unreachable: {exc}") from exc

    def _search(self, bbox: BBox, days: int, max_cloud: float) -> list:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days)
        items = list(
            self._client()
            .search(
                collections=["sentinel-2-l2a"],
                bbox=bbox.as_list(),
                datetime=f"{start:%Y-%m-%d}/{end:%Y-%m-%d}",
                query={"eo:cloud_cover": {"lt": max_cloud}},
                max_items=50,
            )
            .items()
        )
        if not items:
            raise NoSceneFound(f"{self.name}: no scene under {max_cloud}% cloud in {days} days")
        return sorted(items, key=lambda i: i.properties.get("eo:cloud_cover", 100.0))

    def load(self, bbox: BBox, *, days: int, max_cloud: float, grid: int) -> Scene:
        try:
            best = self._search(bbox, days, max_cloud)[0]
            return build_scene(best, self.assets, self.scl_asset, bbox, grid, self.name)
        except NoSceneFound:
            raise
        except Exception as exc:
            raise ProviderError(f"{self.name}: {exc}") from exc

    def load_series(self, bbox: BBox, *, days: int, max_cloud: float, grid: int, n: int) -> list[Scene]:
        """Up to ``n`` least-cloudy scenes, at most one per acquisition date, for compositing."""
        try:
            seen, picked = set(), []
            for it in self._search(bbox, days, max_cloud):
                day = it.datetime.date() if it.datetime else it.id
                if day in seen:
                    continue
                seen.add(day)
                picked.append(it)
                if len(picked) == n:
                    break
            return build_scenes(picked, self.assets, self.scl_asset, bbox, grid, self.name)
        except NoSceneFound:
            raise
        except Exception as exc:
            raise ProviderError(f"{self.name}: {exc}") from exc
