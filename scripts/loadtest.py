"""Load test (stdlib only): ``python scripts/loadtest.py http://localhost:8765 --requests 300 --threads 24``.

Hits /assess with offline synthetic scenes (no imagery traffic) and a mix of bad requests. The service passes if it never
returns a 5xx, overload shows up as clean 429s, and bad input is always a 422.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor


def post(base: str, path: str, body: bytes, timeout: float = 60.0) -> tuple[int, float]:
    t = time.perf_counter()
    req = urllib.request.Request(base + path, data=body, headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            r.read()
            return r.status, time.perf_counter() - t
    except urllib.error.HTTPError as e:
        e.read()
        return e.code, time.perf_counter() - t
    except Exception:  # noqa: BLE001 - a dropped connection counts as a failure, reported as 0
        return 0, time.perf_counter() - t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("--requests", type=int, default=300)
    ap.add_argument("--threads", type=int, default=24)
    a = ap.parse_args()
    rng = random.Random(0)
    bodies = []
    for _ in range(a.requests):
        kind = rng.random()
        if kind < 0.75:
            bodies.append(("/assess", json.dumps({"lat": rng.uniform(25, 48), "lon": rng.uniform(-120, -75), "archetype": "floodplain"}).encode(), "ok"))
        elif kind < 0.9:
            bodies.append(("/assess", json.dumps({"lat": rng.choice([91, -91, "x", None]), "lon": 0}).encode(), "bad"))
        else:
            bodies.append(("/assess/area", json.dumps({"geometry": {"type": "Point", "coordinates": [0, 0]}}).encode(), "bad"))
    t0 = time.perf_counter()
    with ThreadPoolExecutor(a.threads) as pool:
        results = list(pool.map(lambda b: (*post(a.base, b[0], b[1]), b[2]), bodies))
    wall = time.perf_counter() - t0
    codes = Counter(c for c, _, _ in results)
    ok_lat = sorted(d for c, d, k in results if k == "ok" and c == 200)
    server_errors = [c for c, _, _ in results if c >= 500 or c == 0]
    bad_wrong = [c for c, _, k in results if k == "bad" and c not in (422, 429)]
    print(f"{len(results)} requests, {a.threads} threads, {wall:.1f}s ({len(results) / wall:.0f} req/s)")
    print("status codes:", dict(sorted(codes.items())))
    if ok_lat:
        print(f"successful /assess latency: p50 {statistics.median(ok_lat) * 1000:.0f} ms, p95 {ok_lat[int(len(ok_lat) * 0.95) - 1] * 1000:.0f} ms")
    print(f"5xx or dropped: {len(server_errors)}; bad input not answered with 422/429: {len(bad_wrong)}")
    return 1 if server_errors or bad_wrong else 0


if __name__ == "__main__":
    sys.exit(main())
