"""``python -m georisk.validation.grid_collect solar``: grid-proximity features for every sampled point (resumable)."""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from ..grid import GridUnavailable, load_grid


def one(pt: dict) -> dict:
    row = {"lat": pt["lat"], "lon": pt["lon"]}
    try:
        row |= {f"g_{k}": v for k, v in asdict(load_grid(pt["lat"], pt["lon"])).items()}  # the exact function serving uses
    except GridUnavailable as exc:
        row["grid_error"] = str(exc)[:100]
    return row


def collect(points: list[dict], out: Path, workers: int = 16, progress: bool = True) -> int:
    done = set()
    if out.exists():
        done = {(r["lat"], r["lon"]) for r in map(json.loads, out.read_text().splitlines()) if "grid_error" not in r}
    todo = [p for p in points if (p["lat"], p["lon"]) not in done]
    t0, n = time.time(), 0
    with ThreadPoolExecutor(max_workers=workers) as pool, out.open("a") as f:
        for i, row in enumerate(pool.map(one, todo), 1):
            f.write(json.dumps(row) + "\n")
            f.flush()
            n += "grid_error" not in row
            if progress and i % 200 == 0:
                print(f"{i}/{len(todo)} in {time.time() - t0:.0f}s", flush=True)
    return n


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "solar"
    points = json.loads(Path(f"validation_data/{name}.points.json").read_text())
    print(f"{len(points)} points", flush=True)
    print("ok rows:", collect(points, Path("validation_data/grid.jsonl")), flush=True)


if __name__ == "__main__":
    main()
