"""Fetch 6-band, 224x224 (30 m, ~6.7 km) Sentinel-2 chips in the band order Prithvi was trained on."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ..aoi import bbox_from_point
from ..providers._common import BAD_SCL, build_scene  # noqa: F401 - BAD_SCL documents the mask used
from ..providers.earth_search import EarthSearchProvider

log = logging.getLogger(__name__)
CHIP_PX, CHIP_HALF_M = 224, 3360  # 224 px x 30 m = 6.72 km
# Prithvi's HLS bands: blue, green, red, narrow NIR (B8A), SWIR1, SWIR2. "nir" maps to B8A here on purpose.
ASSETS = {"blue": "blue", "green": "green", "red": "red", "nir": "nir08", "swir1": "swir16", "swir2": "swir22"}
ORDER = ("blue", "green", "red", "nir", "swir1", "swir2")
CACHE = Path.home() / ".cache" / "georisk_chips"


def key(lat: float, lon: float) -> str:
    return f"{lat:.5f}_{lon:.5f}"


def fetch_chip(lat: float, lon: float, *, days: int = 240, max_cloud: float = 25.0, min_valid: float = 0.8, tries: int = 3) -> np.ndarray:
    """(6, 224, 224) float32 in reflectance x 10000, clouds filled with the band median. Raises if no clear scene."""
    prov = EarthSearchProvider()
    bbox = bbox_from_point(lat, lon, CHIP_HALF_M)
    last = None
    for item in prov._search(bbox, days, max_cloud)[:tries]:
        try:
            sc = build_scene(item, ASSETS, prov.scl_asset, bbox, CHIP_PX, prov.name)
        except Exception as exc:  # noqa: BLE001 - try the next-clearest scene
            last = exc
            continue
        if sc.valid.mean() < min_valid:
            continue
        out = []
        for b in ORDER:
            arr = sc.bands[b].astype("float32") * 1e4
            fill = float(np.median(arr[sc.valid])) if sc.valid.any() else 0.0
            arr = np.where(sc.valid, arr, fill)
            out.append(arr)
        return np.stack(out).astype("float32")
    raise RuntimeError(f"no clear scene for {lat:.4f},{lon:.4f}" + (f" ({last})" if last else ""))


def collect(points: list[tuple[float, float]], workers: int = 4) -> dict[str, str]:
    """Fetch and cache a chip per point; returns {key: 'ok'|error}. Safe to re-run: cached chips are skipped."""
    CACHE.mkdir(parents=True, exist_ok=True)

    def one(pt):
        k = key(*pt)
        path = CACHE / f"{k}.npy"
        if path.exists():
            return k, "ok"
        try:
            np.save(path, fetch_chip(*pt).astype("float16"))
            return k, "ok"
        except Exception as exc:  # noqa: BLE001 - one bad point must not stop the run
            return k, f"error: {str(exc)[:80]}"

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(one, points))
