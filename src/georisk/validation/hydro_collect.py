"""``python -m georisk.validation.hydro_collect``: compute hydrology features for every labelled flood point (resumable)."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from ..terrain import TerrainUnavailable, load_hydrology

FILES = [f"validation_data/{n}.jsonl" for n in ("flood", "flood2", "flood3", "flood_ohio")]
OUT = Path("validation_data/hydro.jsonl")


def points() -> list[tuple[float, float]]:
    seen, pts = set(), []
    for f in FILES:
        for line in Path(f).read_text().splitlines():
            r = json.loads(line) if line.strip() else None
            if r and (r["lat"], r["lon"]) not in seen:
                seen.add((r["lat"], r["lon"]))
                pts.append((r["lat"], r["lon"]))
    return pts


def one(pt: tuple[float, float]) -> dict:
    lat, lon = pt
    try:
        return {"lat": lat, "lon": lon, **{f"h_{k}": v for k, v in asdict(load_hydrology(lat, lon)).items()}}
    except TerrainUnavailable as exc:
        return {"lat": lat, "lon": lon, "hydro_error": str(exc)[:100]}


def main() -> None:
    done = set()
    if OUT.exists():
        done = {(r["lat"], r["lon"]) for r in map(json.loads, OUT.read_text().splitlines()) if "hydro_error" not in r}
    todo = [p for p in points() if p not in done]
    print(f"{len(todo)} to compute ({len(done)} already done)", flush=True)
    t = time.time()
    with ThreadPoolExecutor(max_workers=6) as pool, OUT.open("a") as f:
        for i, row in enumerate(pool.map(one, todo), 1):
            f.write(json.dumps(row) + "\n")
            f.flush()
            if i % 50 == 0:
                print(f"{i}/{len(todo)} in {time.time() - t:.0f}s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
