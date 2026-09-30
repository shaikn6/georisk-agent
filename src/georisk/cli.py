"""Command line: ``georisk --lat 29.76 --lon -95.37 --radius 800``."""

from __future__ import annotations

import argparse
import json

from .agent import assess
from .synthetic import ARCHETYPES


def main() -> None:
    p = argparse.ArgumentParser(description="Satellite risk screening for a point of interest")
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--radius", type=float, default=500, help="half-side of the square AOI in metres")
    p.add_argument("--days", type=int, default=60)
    p.add_argument("--offline", choices=sorted(ARCHETYPES), help="use a synthetic scene (no network)")
    p.add_argument("--no-llm", action="store_true")
    p.add_argument("--no-terrain", action="store_true", help="skip DEM lookup")
    a = p.parse_args()
    out = assess(a.lat, a.lon, a.radius, archetype=a.offline, days=a.days, use_llm=not a.no_llm, with_terrain=not a.no_terrain)
    print(json.dumps(out.to_dict(), indent=2))


if __name__ == "__main__":
    main()
