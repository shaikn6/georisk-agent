"""Shared Sentinel-2 windowed-read + masking logic used by every STAC-backed provider."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.windows import from_bounds

from ..aoi import BBox
from ..scene import Scene

# SCL classes to reject: nodata, saturated, cloud shadow, cloud (med/high), cirrus, snow
BAD_SCL = (0, 1, 3, 8, 9, 10, 11)
_BASELINE_CUTOFF = datetime(2022, 1, 25, tzinfo=timezone.utc)


# Isolate GDAL from the host environment: public buckets must never depend on (or be broken by)
# whatever AWS credentials happen to be configured on the machine, and no read may hang forever.
_GDAL_ENV = {
    "AWS_NO_SIGN_REQUEST": "YES",
    "GDAL_HTTP_TIMEOUT": "30",
    "GDAL_HTTP_CONNECTTIMEOUT": "10",
    "GDAL_HTTP_MAX_RETRY": "2",
    "GDAL_HTTP_RETRY_DELAY": "1",
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
}


def read_window(href: str, bbox: BBox, out_shape: tuple[int, int], fill_value: float = 0) -> np.ndarray:
    with rasterio.Env(**_GDAL_ENV), rasterio.open(href) as ds:
        to_ds = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
        x0, y0 = to_ds.transform(bbox.west, bbox.south)
        x1, y1 = to_ds.transform(bbox.east, bbox.north)
        win = from_bounds(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1), ds.transform)
        return ds.read(1, window=win, out_shape=out_shape, boundless=True, fill_value=fill_value)


def build_scene(item, assets: dict[str, str], scl_asset: str, bbox: BBox, grid: int, source: str) -> Scene:
    shape = (grid, grid)
    names = list(assets)
    hrefs = [item.assets[scl_asset].href] + [item.assets[assets[n]].href for n in names]
    with ThreadPoolExecutor(max_workers=len(hrefs)) as pool:  # independent COG reads run concurrently
        arrays = list(pool.map(lambda h: read_window(h, bbox, shape), hrefs))
    scl, band_dn = arrays[0], dict(zip(names, arrays[1:], strict=True))
    valid = ~np.isin(scl, BAD_SCL)
    acquired = item.datetime or datetime.fromisoformat(item.properties["datetime"].replace("Z", "+00:00"))
    offset = -0.1 if acquired >= _BASELINE_CUTOFF else 0.0
    bands = {}
    for name, dn in band_dn.items():
        bands[name] = np.clip(dn.astype("float64") * 1e-4 + offset, 0.0, 1.0)
    return Scene(
        bands=bands,
        valid=valid,
        source=f"{source}:{item.id}",
        acquired=acquired.isoformat(),
        cloud_cover=float(item.properties.get("eo:cloud_cover", float("nan"))),
        meta={"platform": item.properties.get("platform"), "grid": grid, "provider": source},
    )
