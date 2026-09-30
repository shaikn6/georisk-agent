"""Live Sentinel-2 L2A loader (Element84 Earth Search STAC + windowed COG reads).

Only the pixels inside the area of interest are fetched, so a 1 km box costs a few
hundred KB rather than a full 100 km tile.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import rasterio
from pyproj import Transformer
from pystac_client import Client
from rasterio.windows import from_bounds

from .aoi import BBox
from .scene import Scene

EARTH_SEARCH = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"
_ASSETS = {"green": "green", "red": "red", "nir": "nir", "swir1": "swir16", "swir2": "swir22"}
# SCL classes to reject: nodata, saturated, cloud shadow, cloud (med/high), cirrus, snow
_BAD_SCL = (0, 1, 3, 8, 9, 10, 11)


class NoSceneFound(RuntimeError):
    pass


def find_best_item(bbox: BBox, days: int = 60, max_cloud: float = 30.0):
    """Least-cloudy recent Sentinel-2 L2A item covering the bbox."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    client = Client.open(EARTH_SEARCH)
    search = client.search(
        collections=[COLLECTION],
        bbox=bbox.as_list(),
        datetime=f"{start:%Y-%m-%d}/{end:%Y-%m-%d}",
        query={"eo:cloud_cover": {"lt": max_cloud}},
        max_items=25,
    )
    items = list(search.items())
    if not items:
        raise NoSceneFound(f"no scene under {max_cloud}% cloud in the last {days} days")
    return min(items, key=lambda i: i.properties.get("eo:cloud_cover", 100.0))


def _read_window(href: str, bbox: BBox, out_shape: tuple[int, int]) -> np.ndarray:
    with rasterio.open(href) as ds:
        to_ds = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
        x0, y0 = to_ds.transform(bbox.west, bbox.south)
        x1, y1 = to_ds.transform(bbox.east, bbox.north)
        win = from_bounds(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1), ds.transform)
        return ds.read(1, window=win, out_shape=out_shape, boundless=True, fill_value=0)


def load_scene(bbox: BBox, days: int = 60, max_cloud: float = 30.0, grid: int = 128) -> Scene:
    item = find_best_item(bbox, days=days, max_cloud=max_cloud)
    shape = (grid, grid)
    scl = _read_window(item.assets["scl"].href, bbox, shape)
    valid = ~np.isin(scl, _BAD_SCL)
    acquired = item.datetime or datetime.fromisoformat(item.properties["datetime"].replace("Z", "+00:00"))
    # Baseline >= 04.00 (from 2022-01-25) adds a -1000 DN offset.
    offset = -0.1 if acquired >= datetime(2022, 1, 25, tzinfo=timezone.utc) else 0.0
    bands = {}
    for name, asset in _ASSETS.items():
        dn = _read_window(item.assets[asset].href, bbox, shape).astype("float64")
        bands[name] = np.clip(dn * 1e-4 + offset, 0.0, 1.0)
    return Scene(
        bands=bands,
        valid=valid,
        source=f"earth-search:{item.id}",
        acquired=acquired.isoformat(),
        cloud_cover=float(item.properties.get("eo:cloud_cover", float("nan"))),
        meta={"platform": item.properties.get("platform"), "grid": grid},
    )
