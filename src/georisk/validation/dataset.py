"""Build a labelled dataset: sample points, fetch labels, compute real-imagery features."""

from __future__ import annotations

import json
import logging
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from ..aoi import bbox_from_point
from ..providers import ProviderChain
from ..scoring import extract_features
from ..terrain import TerrainUnavailable, load_terrain
from .labels import fire_label, flood_label

log = logging.getLogger(__name__)

FLOOD_REGIONS = {  # name: (lat, lon, radius_km)
    "houston": (29.76, -95.37, 30), "new_orleans": (29.95, -90.07, 25), "sacramento": (38.58, -121.49, 30),
    "columbus": (39.96, -83.00, 25), "cedar_rapids": (41.98, -91.67, 20), "miami": (25.77, -80.19, 25),
    "charleston": (32.78, -79.93, 25), "st_louis": (38.63, -90.20, 25), "nashville": (36.16, -86.78, 25),
    "pittsburgh": (40.44, -80.00, 25), "dallas": (32.78, -96.80, 30),
    # Ohio case study
    "cincinnati": (39.10, -84.51, 25), "cleveland": (41.50, -81.69, 25), "dayton": (39.76, -84.19, 22),
    "toledo": (41.65, -83.54, 22), "akron": (41.08, -81.52, 20),
}
OHIO = {"columbus", "cincinnati", "cleveland", "dayton", "toledo", "akron"}
FIRE_REGIONS = {  # name: (lat0, lat1, lon0, lon1)
    "norcal_sierra": (38.5, 40.5, -122.5, -120.5), "socal": (33.8, 34.8, -119.5, -116.5),
    "or_cascades": (43.0, 45.0, -123.0, -121.0), "co_front_range": (38.5, 40.5, -106.0, -104.8),
    "az_central": (33.5, 35.0, -112.5, -110.5), "wa_east_cascades": (47.0, 48.5, -121.5, -119.5),
    "nm_north": (34.5, 36.0, -107.0, -105.5),
}


def _jitter_circle(rng: random.Random, lat: float, lon: float, km: float) -> tuple[float, float]:
    import math

    r = km * math.sqrt(rng.random())
    th = rng.random() * 2 * math.pi
    return lat + (r * math.sin(th)) / 110.574, lon + (r * math.cos(th)) / (111.32 * math.cos(math.radians(lat)))


def sample_labelled_points(task: str, per_class: int, seed: int = 0, max_tries: int = 4000,
                           only: set[str] | None = None) -> list[dict]:
    """Rejection-sample until each class has ``per_class`` points. Unmapped/ambiguous points are dropped."""
    rng = random.Random(seed)
    got: dict[int, list[dict]] = {0: [], 1: []}
    tries = 0
    names = [n for n in (FLOOD_REGIONS if task == "flood" else FIRE_REGIONS) if only is None or n in only]
    while min(len(got[0]), len(got[1])) < per_class and tries < max_tries:
        tries += 1
        region = rng.choice(names)
        if task == "flood":
            la, lo, km = FLOOD_REGIONS[region]
            lat, lon = _jitter_circle(rng, la, lo, km)
        else:
            a, b, c, d = FIRE_REGIONS[region]
            lat, lon = rng.uniform(a, b), rng.uniform(c, d)
        try:
            y = flood_label(lat, lon) if task == "flood" else fire_label(lat, lon)
        except Exception as exc:  # noqa: BLE001 - a flaky label lookup only costs one sample
            log.warning("label lookup failed (%s); skipping point", exc)
            continue
        if y is None or len(got[y]) >= per_class:
            continue
        got[y].append({"task": task, "region": region, "lat": round(lat, 5), "lon": round(lon, 5), "label": y})
    return got[0] + got[1]


def feature_row(pt: dict, chain: ProviderChain, radius_m: float = 500, n_composite: int = 1) -> dict:
    bbox = bbox_from_point(pt["lat"], pt["lon"], radius_m)
    res = chain.load_composite(bbox, n=n_composite) if n_composite > 1 else chain.load(bbox)
    f = extract_features(res.scene)
    row = {**pt, "scene": res.scene.source, "acquired": res.scene.acquired, **asdict(f)}
    try:
        t = load_terrain(bbox)
        row.update({f"t_{k}": v for k, v in asdict(t).items()})
    except TerrainUnavailable as exc:
        row["terrain_error"] = str(exc)
    return row


def collect(points: list[dict], chain: ProviderChain, out: Path, workers: int = 4, **kw) -> int:
    """Append feature rows to a JSONL file, resumable (already-collected points are skipped)."""
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["task"], r["lat"], r["lon"]))
    todo = [p for p in points if (p["task"], p["lat"], p["lon"]) not in done]
    ok = 0
    with ThreadPoolExecutor(workers) as pool, out.open("a") as fh:
        futs = {pool.submit(feature_row, p, chain, **kw): p for p in todo}
        for fu in as_completed(futs):
            try:
                fh.write(json.dumps(fu.result()) + "\n")
                fh.flush()
                ok += 1
            except Exception as exc:  # noqa: BLE001 - lost sample, not a lost run
                log.warning("sample %s failed: %s", futs[fu], exc)
    return ok
