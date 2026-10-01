"""Solar-siting labels from the USGS/LBNL US Large-Scale Solar Photovoltaic Database (USPVDB).

Positives are the centroids of existing utility-scale plants. Negatives are *nearby* non-solar land (3-25 km away, never within
2 km of any plant), so a model must separate solar land from its neighbours instead of just learning "sunny state".
Only elevation-derived features are used downstream: panels are visible in imagery, which would leak the label.
"""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path

import numpy as np

CSV = Path("validation_data/raw/uspvdb_v4.csv")
EXCLUDE_KM = 2.0
NEG_RING_KM = (3.0, 25.0)
LOWER48 = (24.0, 50.0, -125.0, -66.0)  # lat0, lat1, lon0, lon1


def load_plants(path: Path = CSV, min_mw: float = 1.0) -> list[dict]:
    out = []
    for r in csv.DictReader(path.open(encoding="utf-8-sig")):
        try:
            lat, lon, mw = float(r["ylat"]), float(r["xlong"]), float(r["p_cap_ac"])
        except (TypeError, ValueError):
            continue
        if mw >= min_mw and LOWER48[0] <= lat <= LOWER48[1] and LOWER48[2] <= lon <= LOWER48[3]:
            out.append({"lat": lat, "lon": lon, "state": r["p_state"], "mw": mw, "year": r.get("p_year")})
    return out


def haversine_km(lat: float, lon: float, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    p1, p2 = math.radians(lat), np.radians(lats)
    dphi, dlam = p2 - p1, np.radians(lons - lon)
    a = np.sin(dphi / 2) ** 2 + math.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
    return 12742.0 * np.arcsin(np.sqrt(a))


def offset(lat: float, lon: float, km: float, bearing_deg: float) -> tuple[float, float]:
    b = math.radians(bearing_deg)
    return lat + km * math.cos(b) / 110.574, lon + km * math.sin(b) / (111.32 * math.cos(math.radians(lat)))


def sample_points(plants: list[dict], n: int, seed: int = 0) -> list[dict]:
    """n positives (random plants) and n negatives (one nearby non-solar point per sampled plant), keyed by state."""
    rng = random.Random(seed)
    lats, lons = np.array([p["lat"] for p in plants]), np.array([p["lon"] for p in plants])
    chosen = rng.sample(plants, min(n, len(plants)))
    pts = []
    for p in chosen:
        pts.append({"task": "solar", "region": p["state"], "lat": round(p["lat"], 5), "lon": round(p["lon"], 5), "label": 1, "mw": p["mw"]})
        for _ in range(50):  # rejection-sample a nearby point that is clearly not a plant
            la, lo = offset(p["lat"], p["lon"], rng.uniform(*NEG_RING_KM), rng.uniform(0, 360))
            if not (LOWER48[0] <= la <= LOWER48[1] and LOWER48[2] <= lo <= LOWER48[3]):
                continue
            if haversine_km(la, lo, lats, lons).min() >= EXCLUDE_KM:
                pts.append({"task": "solar", "region": p["state"], "lat": round(la, 5), "lon": round(lo, 5), "label": 0, "mw": 0.0})
                break
    return pts
