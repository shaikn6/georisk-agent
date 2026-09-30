"""``python -m georisk.foundation.run``: fetch a chip per labelled point, embed with Prithvi, save the embeddings."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .chips import CACHE, collect, key

FILES = [f"validation_data/{n}.jsonl" for n in ("flood", "flood2", "flood3", "flood_ohio", "fire", "fire2")]
OUT = Path("validation_data/prithvi_embeddings.npz")


def points() -> list[tuple[float, float]]:
    seen, pts = set(), []
    for f in FILES:
        if not Path(f).exists():
            continue
        for line in Path(f).read_text().splitlines():
            r = json.loads(line) if line.strip() else None
            if r and (r["lat"], r["lon"]) not in seen:
                seen.add((r["lat"], r["lon"]))
                pts.append((r["lat"], r["lon"]))
    return pts


def main() -> None:
    pts = points()
    print(f"{len(pts)} unique labelled points", flush=True)
    t = time.time()
    status = collect(pts, workers=4)
    bad = {k: v for k, v in status.items() if v != "ok"}
    print(f"chips: {len(status) - len(bad)} ok, {len(bad)} failed in {time.time() - t:.0f}s", flush=True)
    for k, v in list(bad.items())[:10]:
        print("  ", k, v, flush=True)
    from .embed import PrithviEmbedder

    emb = PrithviEmbedder()
    keys = [key(*p) for p in pts if status.get(key(*p)) == "ok"]
    out = []
    for i in range(0, len(keys), 32):
        chips = np.stack([np.load(CACHE / f"{k}.npy").astype("float32") for k in keys[i : i + 32]])
        out.append(emb.embed(chips))
        print(f"embedded {min(i + 32, len(keys))}/{len(keys)}", flush=True)
    np.savez_compressed(OUT, keys=np.array(keys), emb=np.concatenate(out).astype("float16"))
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
