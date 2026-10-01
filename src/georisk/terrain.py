"""Terrain context from the Copernicus 30 m DEM. Optional: hazards degrade cleanly without it."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .aoi import BBox
from .providers.base import ProviderError

log = logging.getLogger(__name__)
_EARTH_SEARCH = "https://earth-search.aws.element84.com/v1"


class TerrainUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class TerrainFeatures:
    mean_slope_deg: float
    p90_slope_deg: float
    relief_m: float            # max - min elevation in the AOI
    center_percentile: float   # 0 = lowest point of the AOI, 1 = highest
    low_lying_fraction: float  # share of pixels within 2 m of the AOI minimum


def terrain_features(dem: np.ndarray, bbox: BBox) -> TerrainFeatures:
    """Slope and relative-elevation stats from a DEM array covering ``bbox`` (metres)."""
    if dem.ndim != 2 or min(dem.shape) < 3:
        raise ValueError("DEM must be a 2-D array of at least 3x3")
    dem = dem.astype("float64")
    valid = np.isfinite(dem)
    if valid.sum() < 9:
        raise ValueError("DEM has too few valid pixels")
    dem = np.where(valid, dem, np.nanmedian(dem))
    # The Copernicus product is a *surface* model (buildings, canopy). A 3x3 median suppresses
    # single-pixel spikes so towers and tree tops don't masquerade as steep terrain.
    pad = np.pad(dem, 1, mode="edge")
    stack = np.stack([pad[i : i + dem.shape[0], j : j + dem.shape[1]] for i in range(3) for j in range(3)])
    dem = np.median(stack, axis=0)
    mid_lat = np.radians((bbox.north + bbox.south) / 2)
    width_m = (bbox.east - bbox.west) * 111_320 * np.cos(mid_lat)
    height_m = (bbox.north - bbox.south) * 110_574
    dy, dx = height_m / dem.shape[0], width_m / dem.shape[1]
    gy, gx = np.gradient(dem, dy, dx)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    lo, hi = float(dem.min()), float(dem.max())
    cy, cx = dem.shape[0] // 2, dem.shape[1] // 2
    return TerrainFeatures(
        mean_slope_deg=float(slope.mean()),
        p90_slope_deg=float(np.percentile(slope, 90)),
        relief_m=hi - lo,
        center_percentile=float((dem <= dem[cy, cx]).mean()),
        low_lying_fraction=float((dem <= lo + 2.0).mean()),
    )


def combine_first_valid(arrays: list[np.ndarray]) -> np.ndarray:
    """Mosaic co-registered windows: each pixel takes the first tile that actually covers it (non-NaN)."""
    if not arrays:
        raise ValueError("no tiles to combine")
    out = arrays[0].astype("float64").copy()
    for a in arrays[1:]:
        out = np.where(np.isnan(out), a, out)
    return out


def load_terrain(bbox: BBox, grid: int = 64) -> TerrainFeatures:
    """Fetch the DEM window and summarise it. Raises TerrainUnavailable on any failure."""
    try:
        from pystac_client import Client

        from .providers._common import read_window

        items = list(
            Client.open(_EARTH_SEARCH, timeout=20)
            .search(collections=["cop-dem-glo-30"], bbox=bbox.as_list(), max_items=4)
            .items()
        )
        if not items:
            raise ProviderError("no DEM tile covers this AOI")
        # An AOI can straddle 1-degree tiles: read every intersecting tile with NaN outside
        # its footprint, then mosaic, so no pixel is ever invented.
        tiles = [
            read_window(i.assets["data"].href, bbox, (grid, grid), fill_value=np.nan).astype("float64")
            for i in items
        ]
        dem = combine_first_valid(tiles)
        dem[dem <= -1000] = np.nan  # nodata sentinel
        return terrain_features(dem, bbox)
    except Exception as exc:
        raise TerrainUnavailable(str(exc)) from exc


HYDRO_HALF_M, HYDRO_GRID, AOI_HALF_M = 3000, 200, 500  # 6 km window at 30 m; features summarise the central 1 km


def load_dem_features(lat: float, lon: float):
    """(HydroFeatures, TopoFeatures) for the site from one 6 km DEM window. Raises TerrainUnavailable on failure."""
    from .aoi import bbox_from_point
    from .hydrology import hydrology_features, topo_features

    try:
        from pystac_client import Client

        from .providers._common import read_window

        bbox = bbox_from_point(lat, lon, HYDRO_HALF_M)
        items = list(Client.open(_EARTH_SEARCH, timeout=20).search(collections=["cop-dem-glo-30"], bbox=bbox.as_list(), max_items=4).items())
        if not items:
            raise ProviderError("no DEM tile covers this site")
        tiles = [read_window(i.assets["data"].href, bbox, (HYDRO_GRID, HYDRO_GRID), fill_value=np.nan).astype("float64") for i in items]
        dem = combine_first_valid(tiles)
        dem[dem <= -1000] = np.nan
        bad = ~np.isfinite(dem)
        if bad.mean() > 0.05:
            raise ValueError(f"{bad.mean():.0%} of the DEM window is missing")
        dem = np.where(bad, np.nanmedian(dem), dem)
        pad = np.pad(dem, 1, mode="edge")  # same 3x3 median as the terrain features: towers/canopy are not terrain
        dem = np.median(np.stack([pad[i : i + dem.shape[0], j : j + dem.shape[1]] for i in range(3) for j in range(3)]), axis=0)
        cell_m = 2 * HYDRO_HALF_M / HYDRO_GRID
        half = round(AOI_HALF_M / cell_m)
        c = HYDRO_GRID // 2
        aoi = (slice(c - half, c + half), slice(c - half, c + half))
        return hydrology_features(dem, cell_m, aoi), topo_features(dem, cell_m, aoi)
    except Exception as exc:
        raise TerrainUnavailable(str(exc)) from exc


def load_hydrology(lat: float, lon: float):
    """Height-above-nearest-drainage features only (see load_dem_features)."""
    return load_dem_features(lat, lon)[0]
