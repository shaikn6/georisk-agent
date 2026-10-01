"""Area mode: an outline in, per-tile assessments out. Each tile is independent, so one failure never sinks the rest."""

from __future__ import annotations

import logging
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from pyproj import Geod
from shapely.geometry import box, mapping, shape
from shapely.geometry.base import BaseGeometry

from .providers.chain import AllProvidersFailed
from .resultcache import cached_assess as assess

log = logging.getLogger(__name__)
_GEOD = Geod(ellps="WGS84")
MAX_AREA_KM2 = 400.0


class AreaError(ValueError):
    """The outline cannot be assessed (invalid, too large, or too many tiles); the message is safe to show."""


@dataclass(frozen=True)
class Tile:
    lat: float
    lon: float
    bbox: tuple[float, float, float, float]  # west, south, east, north


def parse_geometry(geojson: dict) -> BaseGeometry:
    """Accept a Polygon/MultiPolygon, a Feature, or a one-feature FeatureCollection (WGS84)."""
    try:
        g = geojson
        if g.get("type") == "FeatureCollection":
            feats = g.get("features") or []
            if len(feats) != 1:
                raise AreaError("send exactly one feature")
            g = feats[0]
        if g.get("type") == "Feature":
            g = g.get("geometry") or {}
        if g.get("type") not in ("Polygon", "MultiPolygon"):
            raise AreaError("geometry must be a Polygon or MultiPolygon")
        geom = shape(g)
    except AreaError:
        raise
    except Exception as exc:
        raise AreaError("could not read the GeoJSON geometry") from exc
    if geom.is_empty or not geom.is_valid:
        raise AreaError("geometry is empty or self-intersecting")
    w, s, e, n = geom.bounds
    if not (-180 <= w <= e <= 180 and -90 <= s <= n <= 90):
        raise AreaError("coordinates must be longitude/latitude in degrees")
    area_km2 = abs(_GEOD.geometry_area_perimeter(geom)[0]) / 1e6
    if area_km2 > MAX_AREA_KM2:
        raise AreaError(f"area is {area_km2:.0f} km2; the limit is {MAX_AREA_KM2:.0f} km2")
    return geom


def plan_tiles(geom: BaseGeometry, tile_m: float, max_tiles: int) -> list[Tile]:
    """Square tiles of ``tile_m`` metres laid on a geodesic grid; keep those whose centre is inside the outline."""
    if not 250 <= tile_m <= 5000:
        raise AreaError("tile_m must be between 250 and 5000")
    w, s, e, n = geom.bounds
    mid_lat = (s + n) / 2
    dlat = _GEOD.fwd(0, mid_lat, 0, tile_m)[1] - mid_lat
    dlon = _GEOD.fwd(w, mid_lat, 90, tile_m)[0] - w
    if dlat <= 0 or dlon <= 0:
        raise AreaError("could not size tiles at this location")
    tiles: list[Tile] = []
    rows, cols = math.ceil((n - s) / dlat), math.ceil((e - w) / dlon)
    if rows * cols > max_tiles * 40:  # cheap guard before enumerating a huge empty grid
        raise AreaError("outline too large for this tile size; raise tile_m")
    for r in range(rows):
        for c in range(cols):
            south, west = s + r * dlat, w + c * dlon
            cell = box(west, south, west + dlon, south + dlat)
            centre = cell.centroid
            if geom.contains(centre) or geom.intersects(centre):
                tiles.append(Tile(round(centre.y, 6), round(centre.x, 6), (west, south, west + dlon, south + dlat)))
    if not tiles:
        raise AreaError("no tile centre falls inside the outline; use a larger area or smaller tile_m")
    if len(tiles) > max_tiles:
        raise AreaError(f"{len(tiles)} tiles needed but the limit is {max_tiles}; raise tile_m or draw a smaller area")
    return tiles


def _one(tile: Tile, tile_m: float, **kw) -> dict:
    props: dict = {"lat": tile.lat, "lon": tile.lon}
    try:
        a = assess(tile.lat, tile.lon, tile_m / 2, use_llm=False, **kw).to_dict()
        props |= {"ok": True, "calibrated": {k: v["probability"] for k, v in a["calibrated"].items()},
                  "overall": a["overall"], "confidence": a["confidence"]["level"], "warnings": a["warnings"],
                  "scene": (a.get("acquired") or "")[:10]}
    except AllProvidersFailed:
        props |= {"ok": False, "error": "imagery unavailable"}
    except Exception:
        log.exception("tile failed")  # detail stays in server logs
        props |= {"ok": False, "error": "tile could not be assessed"}
    return props


def assess_area(geojson: dict, tile_m: float = 1000, max_tiles: int = 16, workers: int = 4, **kw) -> dict:
    """GeoJSON FeatureCollection of per-tile results plus a summary. Tiles fail independently."""
    geom = parse_geometry(geojson)
    tiles = plan_tiles(geom, tile_m, max_tiles)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda t: _one(t, tile_m, **kw), tiles))
    feats = [{"type": "Feature", "geometry": mapping(box(*t.bbox)), "properties": r} for t, r in zip(tiles, results, strict=True)]
    ok = [r for r in results if r["ok"]]
    summary: dict = {"tiles": len(tiles), "assessed": len(ok), "failed": len(tiles) - len(ok), "tile_m": tile_m}
    for task in sorted({k for r in ok for k in r["calibrated"]}):
        vals = [r["calibrated"][task] for r in ok if task in r["calibrated"]]
        worst = max((r for r in ok if task in r["calibrated"]), key=lambda r: r["calibrated"][task])
        summary[task] = {"mean": round(sum(vals) / len(vals), 3), "max": max(vals), "tiles_above_50pct": sum(v >= 0.5 for v in vals),
                         "worst_tile": {"lat": worst["lat"], "lon": worst["lon"]}}
    return {"type": "FeatureCollection", "features": feats, "summary": summary}
