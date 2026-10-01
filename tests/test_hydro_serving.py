import dataclasses
import time

import numpy as np

from georisk import agent
from georisk.hydrology import HydroFeatures, TopoFeatures
from georisk.synthetic import make_scene
from georisk.terrain import TerrainFeatures, TerrainUnavailable
from georisk.validation.metrics import Logistic

T = TerrainFeatures(3.0, 6.0, 20.0, 0.5, 0.1)
H = HydroFeatures(2.0, 0.0, 0.5, 100.0, 0.3, 4.0)
O = TopoFeatures(250.0, 1.0, 3.0, 0.2)


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


def test_one_failing_dem_read_only_costs_its_own_features(monkeypatch):
    def boom(*a):
        raise TerrainUnavailable("DEM host down")

    monkeypatch.setattr(agent, "load_terrain", lambda bbox: T)
    monkeypatch.setattr(agent, "load_dem_features", boom)
    terrain, hydro, topo, warns = agent._terrain_context(None, 1.0, 2.0)
    assert terrain == T and hydro is None and topo is None and any("hydrology unavailable" in w for w in warns)
    monkeypatch.setattr(agent, "load_terrain", boom)
    monkeypatch.setattr(agent, "load_dem_features", lambda lat, lon: (H, O))
    terrain, hydro, topo, warns = agent._terrain_context(None, 1.0, 2.0)
    assert terrain is None and hydro == H and topo == O and any("terrain unavailable" in w for w in warns)


def test_the_two_dem_reads_run_together_not_one_after_the_other(monkeypatch):
    def slow_t(bbox):
        time.sleep(0.4)
        return T

    def slow_h(lat, lon):
        time.sleep(0.4)
        return H, O

    monkeypatch.setattr(agent, "load_terrain", slow_t)
    monkeypatch.setattr(agent, "load_dem_features", slow_h)
    t0 = time.time()
    agent._terrain_context(None, 1.0, 2.0)
    assert time.time() - t0 < 0.7  # ~0.4 s in parallel, not 0.8 s in sequence


def test_assess_says_so_when_river_height_data_is_missing(monkeypatch):
    scene = dataclasses.replace(make_scene("floodplain"), source="earth-search:test-scene")  # a "real" (non-synthetic) scene
    monkeypatch.setattr(agent, "_terrain_context", lambda bbox, lat, lon: (T, None, None, []))
    a = agent.assess(39.96, -83.0, 500, scene=scene, use_llm=False)
    assert any("river-height data unavailable" in w for w in a.warnings)
    monkeypatch.setattr(agent, "_terrain_context", lambda bbox, lat, lon: (T, H, O, []))
    b = agent.assess(39.96, -83.0, 500, scene=scene, use_llm=False)
    assert not any("river-height" in w for w in b.warnings) and b.hydrology == H
