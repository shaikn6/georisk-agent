import numpy as np
import pytest

from georisk.aoi import bbox_from_point
from georisk.config import Settings
from georisk.providers import AllProvidersFailed, ProviderChain, ProviderError, SceneCache
from georisk.providers.base import NoSceneFound
from georisk.providers.synthetic import SyntheticProvider
from georisk.synthetic import make_scene

BBOX = bbox_from_point(40.0, -83.0, 500)


class Fake:
    def __init__(self, name, behaviour):
        self.name, self.behaviour, self.calls = name, behaviour, 0

    def probe(self):
        if self.behaviour == "down":
            raise ProviderError("down")

    def load(self, bbox, *, days, max_cloud, grid):
        self.calls += 1
        if self.behaviour == "down":
            raise ProviderError(f"{self.name} exploded")
        if self.behaviour == "empty":
            raise NoSceneFound("nothing recent")
        s = make_scene("healthy_forest")
        s.source = f"{self.name}:scene"
        return s


def chain(providers, tmp_path=None, ttl=3600, **kw):
    cache = SceneCache(tmp_path, ttl) if tmp_path else None
    return ProviderChain(providers, Settings(retry_attempts=2), cache=cache, sleep=lambda s: None, **kw)


def test_primary_used_when_healthy():
    a, b = Fake("a", "ok"), Fake("b", "ok")
    r = chain([a, b]).load(BBOX)
    assert r.scene.source == "a:scene" and b.calls == 0 and not r.degraded


def test_failover_to_second_provider_is_flagged_degraded():
    a, b = Fake("a", "down"), Fake("b", "ok")
    r = chain([a, b]).load(BBOX)
    assert r.scene.source == "b:scene" and r.degraded
    assert a.calls == 2  # retried before giving up on it
    assert r.trail[0]["outcome"].startswith("error") and r.trail[1]["outcome"] == "ok"


def test_no_scene_is_not_retried_and_falls_through():
    a, b = Fake("a", "empty"), Fake("b", "ok")
    r = chain([a, b]).load(BBOX)
    assert a.calls == 1 and r.scene.source == "b:scene"


def test_all_fail_raises_with_full_trail():
    with pytest.raises(AllProvidersFailed) as e:
        chain([Fake("a", "down"), Fake("b", "down")]).load(BBOX)
    assert [t["source"] for t in e.value.trail] == ["a", "b"]


def test_breaker_skips_a_dead_provider_on_later_requests():
    a, b = Fake("a", "down"), Fake("b", "ok")
    c = chain([a, b])
    c.load(BBOX)
    c.load(BBOX)  # second failure opens a's circuit (threshold 2)
    before = a.calls
    r = c.load(BBOX)
    assert a.calls == before and r.trail[0]["outcome"] == "circuit-open"


def test_fresh_cache_hit_skips_all_providers(tmp_path):
    a = Fake("a", "ok")
    c = chain([a], tmp_path)
    c.load(BBOX)
    r = c.load(BBOX)
    assert a.calls == 1 and r.trail[0]["source"] == "cache"


def test_stale_cache_saves_the_request_during_a_full_outage(tmp_path):
    t = {"now": 1_000_000.0}
    cache = SceneCache(tmp_path, ttl_s=60, clock=lambda: t["now"])
    up = ProviderChain([Fake("a", "ok")], Settings(retry_attempts=1), cache=cache, sleep=lambda s: None)
    up.load(BBOX)
    import os
    import time
    old = time.time() - 10_000
    for f in tmp_path.glob("*.npz"):
        os.utime(f, (old, old))
    t["now"] = time.time()
    down = ProviderChain([Fake("a", "down")], Settings(retry_attempts=1), cache=cache, sleep=lambda s: None)
    r = down.load(BBOX)
    assert r.scene.meta["cache"] == "stale" and r.degraded


def test_corrupt_cache_entry_is_discarded_not_fatal(tmp_path):
    a = Fake("a", "ok")
    c = chain([a], tmp_path)
    key = c.cache.key(BBOX, c.settings.lookback_days, c.settings.max_cloud, 128)
    (tmp_path / f"{key}.npz").write_bytes(b"not a real npz")
    r = c.load(BBOX)
    assert r.scene.source == "a:scene" and a.calls == 1


def test_synthetic_fallback_is_last_resort_and_flagged():
    r = chain([Fake("a", "down")], fallback=SyntheticProvider("floodplain")).load(BBOX)
    assert r.scene.source.startswith("synthetic") and r.degraded


def test_cache_roundtrip_preserves_bands(tmp_path):
    cache = SceneCache(tmp_path)
    s = make_scene("dense_urban")
    cache.put("k", s)
    got = cache.get("k")
    assert np.allclose(got.bands["nir"], s.bands["nir"]) and got.source == s.source


def test_empty_provider_list_rejected():
    with pytest.raises(ValueError):
        ProviderChain([])
