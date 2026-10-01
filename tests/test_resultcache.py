import threading
import time

import pytest

from georisk import resultcache
from georisk.resultcache import ResultCache


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_second_call_is_served_from_cache_and_counted():
    c = ResultCache()
    calls = []
    assert c.get_or_compute("k", lambda: calls.append(1) or "v") == "v"
    assert c.get_or_compute("k", lambda: calls.append(1) or "other") == "v" and calls == [1]
    assert (c.hits, c.misses) == (1, 1)


def test_entries_expire_after_the_ttl():
    clock = Clock()
    c = ResultCache(ttl_s=60, clock=clock)
    n = {"i": 0}

    def fn():
        n["i"] += 1
        return n["i"]
    assert c.get_or_compute("k", fn) == 1
    clock.t = 59
    assert c.get_or_compute("k", fn) == 1  # still fresh
    clock.t = 61
    assert c.get_or_compute("k", fn) == 2  # expired: recomputed


def test_cache_is_bounded_and_evicts_the_least_recently_used():
    c = ResultCache(max_entries=2)
    for k in ("a", "b"):
        c.get_or_compute(k, lambda k=k: k)
    c.get_or_compute("a", lambda: "x")  # touch a, so b is now the oldest
    c.get_or_compute("c", lambda: "c")
    assert set(c._data) == {"a", "c"}


def test_failures_are_never_cached_and_do_not_wedge_the_key():
    c = ResultCache()

    def boom():
        raise RuntimeError("provider down")
    with pytest.raises(RuntimeError):
        c.get_or_compute("k", boom)
    assert c.get_or_compute("k", lambda: "recovered") == "recovered"  # next call computes normally
    assert c._inflight == {}


def test_concurrent_identical_requests_do_the_work_once():
    c = ResultCache()
    started, calls = threading.Event(), []

    def slow():
        calls.append(1)
        started.set()
        time.sleep(0.3)
        return "result"
    out = []
    threads = [threading.Thread(target=lambda: out.append(c.get_or_compute("k", slow))) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert out == ["result"] * 6 and len(calls) == 1  # five requests waited for the first instead of repeating it


def test_different_keys_run_in_parallel_not_serialised():
    c = ResultCache()
    t0 = time.time()
    ts = [threading.Thread(target=lambda k=k: c.get_or_compute(k, lambda: time.sleep(0.3))) for k in "abc"]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert time.time() - t0 < 0.7


def test_cached_assess_rounds_coordinates_and_bypasses_for_anything_unusual(monkeypatch):
    resultcache.CACHE.clear()
    calls = []
    monkeypatch.setattr(resultcache, "assess", lambda lat, lon, r, **kw: calls.append((lat, lon, kw)) or object())
    a = resultcache.cached_assess(39.95601, -83.01803, 500)
    b = resultcache.cached_assess(39.95602, -83.01801, 500)  # ~1 m away: same place
    assert a is b and len(calls) == 1
    resultcache.cached_assess(39.9560, -83.0180, 800)  # a different radius is a different question
    resultcache.cached_assess(39.9560, -83.0180, 500, composite=3)
    assert len(calls) == 3
    n = len(calls)
    resultcache.cached_assess(39.9560, -83.0180, 500, archetype="floodplain")  # synthetic: never cached
    resultcache.cached_assess(39.9560, -83.0180, 500, use_llm=True)  # LLM text varies: never cached
    resultcache.cached_assess(39.9560, -83.0180, 500, scene=object())  # injected data: never cached
    assert len(calls) == n + 3
    resultcache.CACHE.clear()
