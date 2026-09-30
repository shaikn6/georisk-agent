import os

import pytest
from fastapi.testclient import TestClient

from georisk.agent import assess
from georisk.api import app

client = TestClient(app)


def test_offline_assessment_end_to_end(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    a = assess(29.76, -95.37, 800, archetype="floodplain", use_llm=False)
    assert a.overall_level in {"moderate", "high", "severe"}
    assert a.narrative_source == "template" and "flood" in a.narrative
    assert a.area_km2 == pytest.approx(2.56, rel=0.03)


def test_api_health_and_assess():
    assert client.get("/health").json() == {"status": "ok"}
    r = client.post("/assess", json={"lat": 34.0, "lon": -118.5, "archetype": "wildland_urban", "use_llm": False})
    assert r.status_code == 200
    body = r.json()
    assert {s["name"] for s in body["scores"]} == {"flood", "wildfire", "urban_density"}


def test_api_validation():
    assert client.post("/assess", json={"lat": 95, "lon": 0}).status_code == 422
    assert client.post("/assess", json={"lat": 0, "lon": 0, "archetype": "nope"}).status_code == 422


@pytest.mark.network
@pytest.mark.skipif(os.environ.get("GEORISK_SKIP_NETWORK") == "1", reason="network disabled")
def test_live_scene_columbus():
    a = assess(39.9612, -82.9988, 500, use_llm=False)
    assert a.source.startswith("earth-search:") and a.features.valid_fraction > 0.3
