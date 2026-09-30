import os

import pytest
from fastapi.testclient import TestClient

from georisk import api
from georisk.agent import assess
from georisk.providers import AllProvidersFailed
from georisk.terrain import TerrainFeatures

client = TestClient(api.app)


def test_offline_assessment_end_to_end(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    a = assess(29.76, -95.37, 800, archetype="floodplain", use_llm=False)
    assert a.overall_level in {"moderate", "high", "severe"}
    assert a.narrative_source == "template" and "flood" in a.narrative
    assert a.provenance["synthetic"] is True and a.confidence.level == "none"
    assert a.area_km2 == pytest.approx(2.56, rel=0.03)


def test_terrain_flows_through_and_is_reported():
    t = TerrainFeatures(0.5, 1.0, 3.0, 0.05, 0.9)
    a = assess(29.76, -95.37, 800, archetype="floodplain", terrain=t, use_llm=False)
    assert a.terrain == t and a.provenance["terrain"] is True
    assert {h.status for h in a.hazards if h.name == "flood"} == {"ok"}


def test_health_is_liveness_only():
    assert client.get("/health").json() == {"status": "ok"}


def test_deep_health_reports_degraded_when_every_provider_is_down(monkeypatch):
    class Chain:
        def health(self):
            return {"a": {"reachable": False, "circuit": "open", "error": "x"}}

    monkeypatch.setattr(api, "default_chain", lambda: Chain())
    body = client.get("/health/deep").json()
    assert body["status"] == "degraded" and body["providers"]["a"]["circuit"] == "open"


def test_api_assess_offline_and_validation():
    r = client.post("/assess", json={"lat": 34.0, "lon": -118.5, "archetype": "wildland_urban", "use_llm": False})
    assert r.status_code == 200 and {s["name"] for s in r.json()["scores"]} == {"flood", "wildfire", "urban_density"}
    assert client.post("/assess", json={"lat": 95, "lon": 0}).status_code == 422
    assert client.post("/assess", json={"lat": 0, "lon": 0, "archetype": "nope"}).status_code == 422


def test_total_imagery_outage_returns_503_with_trail(monkeypatch):
    def boom(*a, **k):
        raise AllProvidersFailed([{"source": "earth-search", "outcome": "error: down"}])

    monkeypatch.setattr(api, "assess", boom)
    r = client.post("/assess", json={"lat": 40.0, "lon": -83.0, "use_llm": False})
    assert r.status_code == 503 and r.json()["detail"]["trail"][0]["source"] == "earth-search"


def test_batch_isolates_failures():
    good = {"lat": 34.0, "lon": -118.5, "archetype": "wildland_urban", "use_llm": False}
    bad = {"lat": 34.0, "lon": -118.5, "archetype": "nope"}
    r = client.post("/assess/batch", json={"items": [good, bad, good]}).json()
    assert r["count"] == 3 and r["succeeded"] == 2
    assert [x["ok"] for x in r["results"]] == [True, False, True]
    assert "archetype" in r["results"][1]["error"]


def test_batch_size_is_bounded():
    item = {"lat": 0, "lon": 0, "archetype": "floodplain"}
    assert client.post("/assess/batch", json={"items": [item] * 26}).status_code == 422
    assert client.post("/assess/batch", json={"items": []}).status_code == 422


@pytest.mark.network
@pytest.mark.skipif(os.environ.get("GEORISK_SKIP_NETWORK") == "1", reason="network disabled")
def test_live_scene_columbus_with_terrain(tmp_path, monkeypatch):
    monkeypatch.setenv("GEORISK_CACHE_DIR", str(tmp_path))
    a = assess(39.9612, -82.9988, 500, use_llm=False)
    assert a.source.split(":")[0] in {"earth-search", "planetary-computer"}
    assert a.features.valid_fraction > 0.3 and a.confidence.score > 0.3


@pytest.mark.network
@pytest.mark.skipif(os.environ.get("GEORISK_SKIP_NETWORK") == "1", reason="network disabled")
def test_dem_tile_boundary_has_no_phantom_relief():
    """Columbus sits on the W083 tile edge; a single-tile read used to fabricate ~240 m of relief."""
    from georisk.aoi import bbox_from_point
    from georisk.terrain import load_terrain

    t = load_terrain(bbox_from_point(39.9612, -82.9988, 800))
    assert t.relief_m < 80 and t.mean_slope_deg < 6
