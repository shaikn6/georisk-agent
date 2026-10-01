"""Proximity to the high-voltage grid (HIFLD transmission lines, >= 100 kV): the practical limit on where utility-scale solar and
large loads such as data centres can connect. Training and live serving share ``features_from_lines`` so they cannot drift apart."""

from __future__ import annotations

import json
import math
import urllib.parse
import urllib.request
from dataclasses import dataclass

import numpy as np
from pyproj import Proj
from shapely.geometry import LineString, Point

SERVICE = "https://services1.arcgis.com/Hp6G80Pky0om7QvQ/arcgis/rest/services/Electric_Power_Transmission_Lines/FeatureServer/0/query"
RADIUS_KM = 25.0  # nothing beyond this is looked at; distances are capped here
_CLASS_KV = {"100-161": 115.0, "220-287": 230.0, "345": 345.0, "500": 500.0, "735 AND ABOVE": 765.0, "DC": 300.0}


class GridUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class GridFeatures:
    dist_km: float         # to the nearest line of 100 kV or more
    dist_230_km: float     # to the nearest line of 230 kV or more (strong interconnection)
    n_lines_10km: float    # line segments within 10 km (a corridor, not a lone spur)
    max_kv_10km: float     # highest voltage within 10 km (0 if none)


def voltage_kv(voltage, volt_class) -> float:
    v = voltage if isinstance(voltage, int | float) and voltage > 0 else None
    return float(v) if v else _CLASS_KV.get(str(volt_class or "").strip().upper(), 115.0)  # unknown voltage: assume the common class


def features_from_lines(lat: float, lon: float, lines: list[tuple[float, np.ndarray]]) -> GridFeatures:
    """lines: [(kV, array of [lon, lat] vertices)]. Distances are metres-accurate in an azimuthal projection centred on the site."""
    proj = Proj(proj="aeqd", lat_0=lat, lon_0=lon, datum="WGS84")
    here, best, best230, near, maxkv = Point(0, 0), RADIUS_KM, RADIUS_KM, 0, 0.0
    for kv, xy in lines:
        if len(xy) < 2:
            continue
        x, y = proj(xy[:, 0], xy[:, 1])
        d = here.distance(LineString(np.column_stack([x, y]))) / 1000.0
        if d > RADIUS_KM:
            continue
        best = min(best, d)
        if kv >= 230:
            best230 = min(best230, d)
        if d <= 10.0:
            near += 1
            maxkv = max(maxkv, kv)
    return GridFeatures(round(best, 3), round(best230, 3), float(near), maxkv)


def _parse(features: list[dict]) -> list[tuple[float, np.ndarray]]:
    out = []
    for f in features:
        a = f.get("attributes", {})
        for path in (f.get("geometry") or {}).get("paths", []):
            if len(path) >= 2:
                out.append((voltage_kv(a.get("VOLTAGE"), a.get("VOLT_CLASS")), np.asarray(path, dtype="float64")))
    return out


def fetch_lines_near(lat: float, lon: float, radius_km: float = RADIUS_KM, timeout: float = 12.0) -> list[tuple[float, np.ndarray]]:
    dlat = radius_km / 110.574
    dlon = radius_km / (111.32 * max(math.cos(math.radians(lat)), 0.2))
    params = {"where": "1=1", "geometry": f"{lon - dlon},{lat - dlat},{lon + dlon},{lat + dlat}", "geometryType": "esriGeometryEnvelope",
              "inSR": 4326, "spatialRel": "esriSpatialRelIntersects", "outFields": "VOLTAGE,VOLT_CLASS", "returnGeometry": "true",
              "outSR": 4326, "geometryPrecision": 4, "resultRecordCount": 2000, "f": "json"}
    try:
        req = urllib.request.Request(SERVICE + "?" + urllib.parse.urlencode(params), headers={"User-Agent": "georisk-agent"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.load(r)
    except Exception as exc:
        raise GridUnavailable(f"grid service unreachable: {exc}") from exc
    if "error" in data:
        raise GridUnavailable(str(data["error"].get("message", "grid service error")))
    return _parse(data.get("features", []))


def load_grid(lat: float, lon: float) -> GridFeatures:
    return features_from_lines(lat, lon, fetch_lines_near(lat, lon))
