"""Area-of-interest helpers."""

from __future__ import annotations

from dataclasses import dataclass

from pyproj import Geod

_GEOD = Geod(ellps="WGS84")


@dataclass(frozen=True)
class BBox:
    """WGS84 bounding box in degrees (west, south, east, north)."""

    west: float
    south: float
    east: float
    north: float

    def as_list(self) -> list[float]:
        return [self.west, self.south, self.east, self.north]

    def area_km2(self) -> float:
        area, _ = _GEOD.polygon_area_perimeter(
            [self.west, self.east, self.east, self.west],
            [self.south, self.south, self.north, self.north],
        )
        return abs(area) / 1e6


def bbox_from_point(lat: float, lon: float, radius_m: float) -> BBox:
    """Square box whose half-side is ``radius_m`` metres, measured geodesically."""
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError(f"invalid coordinate: lat={lat}, lon={lon}")
    if not (10 <= radius_m <= 20_000):
        raise ValueError("radius_m must be between 10 m and 20 km")
    east_lon, _, _ = _GEOD.fwd(lon, lat, 90, radius_m)
    west_lon, _, _ = _GEOD.fwd(lon, lat, 270, radius_m)
    _, north_lat, _ = _GEOD.fwd(lon, lat, 0, radius_m)
    _, south_lat, _ = _GEOD.fwd(lon, lat, 180, radius_m)
    return BBox(west_lon, south_lat, east_lon, north_lat)
