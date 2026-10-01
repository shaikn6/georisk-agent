import dataclasses
import time

import numpy as np

from georisk import agent
from georisk.grid import GridFeatures, GridUnavailable
from georisk.hydrology import HydroFeatures, TopoFeatures
from georisk.synthetic import make_scene
from georisk.terrain import TerrainFeatures, TerrainUnavailable
from georisk.validation.metrics import Logistic

T = TerrainFeatures(3.0, 6.0, 20.0, 0.5, 0.1)
H = HydroFeatures(2.0, 0.0, 0.5, 100.0, 0.3, 4.0)
O = TopoFeatures(250.0, 1.0, 3.0, 0.2, 0.9)
G = GridFeatures(4.0, 9.0, 3.0, 345.0)


def test_input_cap_stops_linear_extrapolation_in_the_tails():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 2))
    y = (X[:, 0] + 0.3 * rng.normal(size=400) > 0).astype(int)
    capped, free = Logistic(1.0, zmax=2.0).fit(X, y), Logistic(1.0).fit(X, y)
    far = np.array([[9.0, 0.0]])  # nine standard deviations out
    assert np.allclose(capped.predict_proba(far), capped.predict_proba(np.array([[2.0, 0.0]])))  # same as at the cap
    assert free.predict_proba(far)[0] > capped.predict_proba(far)[0]  # uncapped saturates harder


def test_old_saved_models_without_a_cap_still_load_and_predict_identically():
    X = np.random.default_rng(1).normal(size=(100, 2))
    y = (X[:, 0] > 0).astype(int)
    m = Logistic(1.0).fit(X, y)
    legacy = {k: v for k, v in m.to_dict().items() if k != "zmax"}  # what a pre-cap artifact looks like
    back = Logistic.from_dict(legacy)
    assert back.zmax is None and np.allclose(back.predict_proba(X), m.predict_proba(X))


def boom_grid(lat, lon):
    raise GridUnavailable("grid host down")


def test_one_failing_dem_read_only_costs_its_own_features(monkeypatch):
    def boom(*a):
        raise TerrainUnavailable("DEM host down")

    monkeypatch.setattr(agent, "load_terrain", lambda bbox: T)
    monkeypatch.setattr(agent, "load_grid", lambda lat, lon: G)
    monkeypatch.setattr(agent, "load_dem_features", boom)
    terrain, hydro, topo, grid, warns = agent._terrain_context(None, 1.0, 2.0)
    assert terrain == T and hydro is None and topo is None and grid == G and any("hydrology unavailable" in w for w in warns)
    monkeypatch.setattr(agent, "load_terrain", boom)
    monkeypatch.setattr(agent, "load_dem_features", lambda lat, lon: (H, O))
    monkeypatch.setattr(agent, "load_grid", boom_grid)
    terrain, hydro, topo, grid, warns = agent._terrain_context(None, 1.0, 2.0)
    assert terrain is None and hydro == H and topo == O and grid is None and any("grid unavailable" in w for w in warns) and any("terrain unavailable" in w for w in warns)


def test_the_two_dem_reads_run_together_not_one_after_the_other(monkeypatch):
    def slow_t(bbox):
        time.sleep(0.4)
        return T

    def slow_h(lat, lon):
        time.sleep(0.4)
        return H, O

    monkeypatch.setattr(agent, "load_terrain", slow_t)
    monkeypatch.setattr(agent, "load_dem_features", slow_h)
    monkeypatch.setattr(agent, "load_grid", lambda lat, lon: G)
    t0 = time.time()
    agent._terrain_context(None, 1.0, 2.0)
    assert time.time() - t0 < 0.7  # ~0.4 s in parallel, not 0.8 s in sequence


def test_assess_says_so_when_river_height_data_is_missing(monkeypatch):
    scene = dataclasses.replace(make_scene("floodplain"), source="earth-search:test-scene")  # a "real" (non-synthetic) scene
    monkeypatch.setattr(agent, "_terrain_context", lambda bbox, lat, lon: (T, None, None, None, []))
    a = agent.assess(39.96, -83.0, 500, scene=scene, use_llm=False)
    assert any("river-height data unavailable" in w for w in a.warnings)
    monkeypatch.setattr(agent, "_terrain_context", lambda bbox, lat, lon: (T, H, O, G, []))
    b = agent.assess(39.96, -83.0, 500, scene=scene, use_llm=False)
    assert not any("river-height" in w for w in b.warnings) and b.hydrology == H


class NoImagery:
    """A provider chain with no usable scene (overcast for the whole search window)."""

    def load(self, bbox, days=None, **kw):
        from georisk.providers.chain import AllProvidersFailed

        raise AllProvidersFailed([{"source": "earth-search", "outcome": "no-scene"}])

    load_composite = load


def test_imagery_outage_still_returns_the_results_that_do_not_need_imagery(monkeypatch):
    monkeypatch.setattr(agent, "_terrain_context", lambda bbox, lat, lon: (T, H, O, G, []))
    a = agent.assess(39.96, -83.0, 500, chain=NoImagery(), use_llm=False)
    assert a.features is None and a.hazards == [] and a.overall is None and a.confidence.level == "low" and a.source == "none"
    assert set(a.calibrated) == {"solar"}  # flood and fire need imagery; solar uses only elevation and grid data
    assert a.calibrated["solar"]["kind"] == "opportunity" and 0 <= a.calibrated["solar"]["probability"] <= 1
    assert any("imagery unavailable" in w for w in a.warnings) and a.provenance["imagery"] is False
    assert a.narrative.startswith("Energy lens") and "imagery unavailable" in a.narrative.lower()
    assert a.to_dict()["hazards"] == []  # still serialisable for the API


def test_nothing_available_at_all_is_still_a_hard_failure(monkeypatch):
    import pytest

    from georisk.providers.chain import AllProvidersFailed

    monkeypatch.setattr(agent, "_terrain_context", lambda bbox, lat, lon: (None, None, None, None, ["terrain unavailable: x"]))
    with pytest.raises(AllProvidersFailed):
        agent.assess(39.96, -83.0, 500, chain=NoImagery(), use_llm=False)


def test_without_terrain_an_imagery_outage_is_still_an_error(monkeypatch):
    import pytest

    from georisk.providers.chain import AllProvidersFailed

    with pytest.raises(AllProvidersFailed):
        agent.assess(39.96, -83.0, 500, chain=NoImagery(), with_terrain=False, use_llm=False)


def test_the_api_serves_the_partial_result_with_200_not_503(monkeypatch):
    from fastapi.testclient import TestClient

    from georisk import api

    api._HITS.clear()
    monkeypatch.setenv("GEORISK_RATE_PER_MIN", "1000")
    monkeypatch.setattr(agent, "_terrain_context", lambda bbox, lat, lon: (T, H, O, G, []))
    monkeypatch.setattr(agent, "default_chain", lambda: NoImagery())
    from georisk import resultcache

    resultcache.CACHE.clear()
    r = TestClient(api.app).post("/assess", json={"lat": 39.97, "lon": -83.01})
    assert r.status_code == 200 and set(r.json()["calibrated"]) == {"solar"} and r.json()["overall"] is None
    resultcache.CACHE.clear()
