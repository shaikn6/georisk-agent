"""Ground-truth labels from public, keyless services. Every call is retried and time-bounded;
an unavailable label means the sample is *skipped*, never guessed."""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request

from ..resilience import retry

log = logging.getLogger(__name__)
FEMA = "https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer/28/query"
_FIRE_BASE = "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services"
FIRE_LAYERS = (
    "InteragencyFirePerimeterHistor_2000s_Read_Only",
    "InteragencyFirePerimeterHistory_2010s_Read_Only",
    "InterAgencyFirePerimeterHistory_2020s_Read_Only",
)


class LabelUnavailable(RuntimeError):
    pass


def _get(url: str, params: dict, timeout: float = 25.0) -> dict:
    def call() -> dict:
        req = urllib.request.Request(url + "?" + urllib.parse.urlencode(params), headers={"User-Agent": "georisk-agent"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.load(r)
        if "error" in data:
            raise LabelUnavailable(str(data["error"].get("message")))
        return data

    return retry(call, attempts=3, base_delay=1.0, never_retry=(LabelUnavailable,))


def flood_label(lat: float, lon: float) -> int | None:
    """1 if the point is in a FEMA Special Flood Hazard Area, 0 if mapped outside one, None if unmapped."""
    d = _get(FEMA, {"geometry": f"{lon},{lat}", "geometryType": "esriGeometryPoint", "inSR": 4326,
                    "spatialRel": "esriSpatialRelIntersects", "outFields": "FLD_ZONE,SFHA_TF",
                    "returnGeometry": "false", "f": "json"})
    feats = d.get("features", [])
    if not feats:
        return None
    return 1 if any(f["attributes"].get("SFHA_TF") == "T" for f in feats) else 0


def _fire_count(where: str, lat: float, lon: float, pad_deg: float = 0.0) -> int:
    total = 0
    for layer in FIRE_LAYERS:
        geom = (
            {"geometry": f"{lon},{lat}", "geometryType": "esriGeometryPoint"}
            if pad_deg == 0
            else {"geometry": f"{lon - pad_deg},{lat - pad_deg},{lon + pad_deg},{lat + pad_deg}",
                  "geometryType": "esriGeometryEnvelope"}
        )
        d = _get(f"{_FIRE_BASE}/{layer}/FeatureServer/0/query",
                 {**geom, "inSR": 4326, "spatialRel": "esriSpatialRelIntersects", "where": where,
                  "returnCountOnly": "true", "f": "json"})
        total += int(d.get("count", 0))
    return total


def fire_label(lat: float, lon: float, y0: int = 2005, y1: int = 2018) -> int | None:
    """1: inside a >=100-acre fire perimeter from y0..y1. 0: no perimeter of any size/year nearby. None: ambiguous."""
    burned = _fire_count(f"FIRE_YEAR_INT>={y0} AND FIRE_YEAR_INT<={y1} AND GIS_ACRES>=100", lat, lon)
    if burned:
        return 1
    anywhere = _fire_count("1=1", lat, lon, pad_deg=0.006)  # ~650 m envelope, any year, any size
    return 0 if anywhere == 0 else None
