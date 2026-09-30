"""``python -m georisk.validation.run --task flood --per-class 50 --out validation_data/flood.jsonl``"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from ..config import Settings
from ..providers import build_default_chain
from .dataset import OHIO, collect, sample_labelled_points


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["flood", "fire"], required=True)
    p.add_argument("--per-class", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--ohio", action="store_true", help="sample only the Ohio case-study cities")
    p.add_argument("--composite", type=int, default=1)
    a = p.parse_args()
    logging.basicConfig(level=logging.WARNING)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    pts_file = a.out.with_suffix(".points.json")
    if pts_file.exists():
        pts = json.loads(pts_file.read_text())
    else:
        pts = sample_labelled_points(a.task, a.per_class, a.seed, only=OHIO if a.ohio else None)
        pts_file.write_text(json.dumps(pts))
    print(f"{a.task}: {len(pts)} labelled points ({sum(p['label'] for p in pts)} positive)", flush=True)
    chain = build_default_chain(Settings())
    n = collect(pts, chain, a.out, workers=a.workers, n_composite=a.composite)
    print(f"{a.task}: collected {n} new rows -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
