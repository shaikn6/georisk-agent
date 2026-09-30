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


def test_composite_median_rejects_outliers_and_unions_validity():
    import numpy as np

    from georisk.composite import median_composite
    from georisk.scene import Scene

    base = make_scene("healthy_forest")
    a = Scene(base.bands, base.valid.copy(), "x:a", "2026-09-01T00:00:00+00:00", 5.0)
    b = Scene(base.bands, base.valid.copy(), "x:b", "2026-09-10T00:00:00+00:00", 3.0)
    cloudy = {k: np.clip(v + 0.5, 0, 1) for k, v in base.bands.items()}  # a haze/cloud outlier
    c = Scene(cloudy, base.valid.copy(), "x:c", "2026-09-20T00:00:00+00:00", 8.0)
    a.valid[:8, :8] = False  # a cannot see this corner...
    comp = median_composite([a, b, c])
    assert np.allclose(comp.bands["nir"][20:, 20:], base.bands["nir"][20:, 20:], atol=1e-9)  # outlier voted out
    assert comp.valid[:8, :8].all()  # ...but b and c can, so the composite can
    assert comp.meta["n_scenes"] == 3 and comp.acquired.startswith("2026-09-20")


def test_composite_of_one_and_error_cases():
    import pytest

    from georisk.composite import median_composite

    assert median_composite([make_scene("floodplain")]).meta["n_scenes"] == 1
    with pytest.raises(ValueError):
        median_composite([])


class SeriesFake(Fake):
    def load_series(self, bbox, *, days, max_cloud, grid, n):
        if self.behaviour == "down":
            raise ProviderError("series down")
        return [make_scene("healthy_forest", seed=i) for i in range(n)]


def test_chain_composite_uses_series_and_degrades_to_single_scene():
    a = SeriesFake("a", "ok")
    r = chain([a]).load_composite(BBOX, n=3)
    assert r.scene.meta["n_scenes"] == 3 and r.trail[0]["scenes"] == 3
    # provider without series support still answers (single scene), so compositing never lowers availability
    r2 = chain([Fake("plain", "ok")]).load_composite(BBOX, n=3)
    assert r2.scene.source == "plain:scene"
    # series source down -> falls through to the single-scene path of another source
    r3 = chain([SeriesFake("a", "down"), Fake("b", "ok")]).load_composite(BBOX, n=3)
    assert r3.scene.source == "b:scene" and r3.degraded
