"""Bounded, expiring, single-flight cache for finished assessments.

Imagery is days old, so an assessment of the same place is stable for minutes. Caching it makes repeat clicks instant and
stops identical requests hammering the imagery providers. Failures are never cached, and two simultaneous requests for
the same place do the work once (the second waits for the first).
"""

from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

from .agent import Assessment, assess


class ResultCache:
    def __init__(self, max_entries: int = 256, ttl_s: float = 1800.0, clock: Callable[[], float] = time.monotonic) -> None:
        self.max_entries, self.ttl_s, self.clock = max_entries, ttl_s, clock
        self._data: OrderedDict[Any, tuple[float, Any]] = OrderedDict()
        self._inflight: dict[Any, threading.Lock] = {}
        self._lock = threading.Lock()
        self.hits = self.misses = 0

    def _fresh(self, key) -> Any | None:
        item = self._data.get(key)
        if item is None:
            return None
        if self.clock() - item[0] > self.ttl_s:
            del self._data[key]
            return None
        self._data.move_to_end(key)
        return item[1]

    def get_or_compute(self, key, fn: Callable[[], Any]) -> Any:
        with self._lock:
            hit = self._fresh(key)
            if hit is not None:
                self.hits += 1
                return hit
            gate = self._inflight.setdefault(key, threading.Lock())
        with gate:  # single flight: concurrent identical requests wait here, then find the first one's result
            with self._lock:
                hit = self._fresh(key)
                if hit is not None:
                    self.hits += 1
                    return hit
                self.misses += 1
            try:
                value = fn()  # an exception propagates and nothing is stored
            finally:
                with self._lock:
                    self._inflight.pop(key, None)
            with self._lock:
                self._data[key] = (self.clock(), value)
                while len(self._data) > self.max_entries:
                    self._data.popitem(last=False)
            return value

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self.hits = self.misses = 0


CACHE = ResultCache(ttl_s=float(os.environ.get("GEORISK_RESULT_TTL_S", "1800")))


def cached_assess(lat: float, lon: float, radius_m: float = 500, *, use_llm: bool = False, **kw) -> Assessment:
    """assess() with caching for the plain real-imagery path. Anything unusual (injected scene, chain, LLM) bypasses it."""
    if use_llm or any(kw.get(k) is not None for k in ("scene", "chain", "terrain")) or kw.get("archetype"):
        return assess(lat, lon, radius_m, use_llm=use_llm, **kw)
    key = (round(lat, 4), round(lon, 4), float(radius_m), kw.get("composite", 1), kw.get("days"), kw.get("with_terrain", True))
    return CACHE.get_or_compute(key, lambda: assess(lat, lon, radius_m, use_llm=False, **kw))
