"""``python -m georisk.validation.dem_collect solar``: elevation-derived features for every sampled point (resumable).

Writes validation_data/<name>.jsonl with the 1 km terrain features (t_*), river height (h_*) and topography (o_*).
Imagery is deliberately not used: for solar siting it would reveal the panels themselves and leak the label.
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from ..aoi import bbox_from_point
from ..terrain import TerrainUnavailable, load_dem_features, load_terrain


def one(pt: dict) -> dict:
    row = dict(pt)
    try:
        t = load_terrain(bbox_from_point(pt["lat"], pt["lon"], 500))
        hydro, topo = load_dem_features(pt["lat"], pt["lon"])
        row |= {f"t_{k}": v for k, v in asdict(t).items()} | {f"h_{k}": v for k, v in asdict(hydro).items()}
        row |= {f"o_{k}": v for k, v in asdict(topo).items()}
    except TerrainUnavailable as exc:
        row["terrain_error"] = str(exc)[:100]
    return row


def collect(points: list[dict], out: Path, workers: int = 8, progress: bool = True) -> int:
    done = set()
    if out.exists():
        done = {(r["lat"], r["lon"]) for r in map(json.loads, out.read_text().splitlines()) if "terrain_error" not in r}
    todo = [p for p in points if (p["lat"], p["lon"]) not in done]
    t0, n = time.time(), 0
    with ThreadPoolExecutor(max_workers=workers) as pool, out.open("a") as f:
        for i, row in enumerate(pool.map(one, todo), 1):
            f.write(json.dumps(row) + "\n")
            f.flush()
            n += "terrain_error" not in row
            if progress and i % 100 == 0:
                print(f"{i}/{len(todo)} in {time.time() - t0:.0f}s", flush=True)
    return n


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "solar"
    points = json.loads(Path(f"validation_data/{name}.points.json").read_text())
    print(f"{len(points)} points", flush=True)
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8  # network-bound: more threads help on scattered points
    print("ok rows:", collect(points, Path(f"validation_data/{name}.jsonl"), workers=workers), flush=True)


if __name__ == "__main__":
    main()
