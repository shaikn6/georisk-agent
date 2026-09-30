import pytest
from fastapi.testclient import TestClient

from georisk import api, area
from georisk.providers.chain import AllProvidersFailed

# ~3 km x 2 km rectangle near Columbus, OH
RECT = {"type": "Polygon", "coordinates": [[[-83.03, 39.95], [-83.0, 39.95], [-83.0, 39.97], [-83.03, 39.97], [-83.03, 39.95]]]}


class _Fake:
    def __init__(self, lat, lon):
        self.lat, self.lon = lat, lon

    def to_dict(self):
        return {"calibrated": {"flood": {"probability": round(0.2 + (self.lon + 83.03) * 10, 3)}}, "overall": 30.0,
                "confidence": {"level": "high"}, "warnings": [], "acquired": "2026-09-25T00:00:00Z"}


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    api._HITS.clear()
    monkeypatch.setenv("GEORISK_RATE_PER_MIN", "1000")
    monkeypatch.setattr(area, "assess", lambda lat, lon, r, **k: _Fake(lat, lon))


def test_tiles_cover_the_outline_with_centres_inside():
    tiles = area.plan_tiles(area.parse_geometry(RECT), 1000, 50)
    assert 4 <= len(tiles) <= 9
    assert all(-83.03 <= t.lon <= -83.0 and 39.95 <= t.lat <= 39.97 for t in tiles)


def test_l_shaped_outline_skips_tiles_outside_it():
    l_shape = {"type": "Polygon", "coordinates": [[[-83.05, 39.95], [-83.0, 39.95], [-83.0, 39.97], [-83.03, 39.97], [-83.03, 40.0], [-83.05, 40.0], [-83.05, 39.95]]]}
    full_bbox = area.plan_tiles(area.parse_geometry({"type": "Polygon", "coordinates": [[[-83.05, 39.95], [-83.0, 39.95], [-83.0, 40.0], [-83.05, 40.0], [-83.05, 39.95]]]}), 1000, 100)
    l_tiles = area.plan_tiles(area.parse_geometry(l_shape), 1000, 100)
    assert len(l_tiles) < len(full_bbox)


@pytest.mark.parametrize("bad", [
    {"type": "Point", "coordinates": [0, 0]},
    {"type": "Polygon", "coordinates": [[[0, 0], [1, 1], [1, 0], [0, 1], [0, 0]]]},  # bow-tie (self-intersecting)
    {"type": "Polygon", "coordinates": []},
    {"type": "FeatureCollection", "features": []},
    {"type": "Polygon", "coordinates": [[[-200, 0], [-199, 0], [-199, 1], [-200, 1], [-200, 0]]]},
    {"hello": "world"},
])
def test_bad_geometry_is_rejected_with_a_clear_error(bad):
    with pytest.raises(area.AreaError):
        area.parse_geometry(bad)


def test_huge_area_and_too_many_tiles_are_refused():
    big = {"type": "Polygon", "coordinates": [[[-90, 35], [-80, 35], [-80, 45], [-90, 45], [-90, 35]]]}
    with pytest.raises(area.AreaError, match="limit"):
        area.parse_geometry(big)
    with pytest.raises(area.AreaError, match="limit"):
        area.plan_tiles(area.parse_geometry(RECT), 250, 10)


def test_one_failing_tile_does_not_sink_the_others(monkeypatch):
    calls = {"n": 0}

    def flaky(lat, lon, r, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            raise AllProvidersFailed([])
        if calls["n"] == 3:
            raise RuntimeError("secret internal detail /Users/x")
        return _Fake(lat, lon)

    monkeypatch.setattr(area, "assess", flaky)
    out = area.assess_area(RECT, tile_m=1000, max_tiles=50, workers=1)
    ok = [f for f in out["features"] if f["properties"]["ok"]]
    bad = [f["properties"] for f in out["features"] if not f["properties"]["ok"]]
    assert len(ok) == len(out["features"]) - 2 and out["summary"]["failed"] == 2
    assert {b["error"] for b in bad} == {"imagery unavailable", "tile could not be assessed"}
    assert "secret" not in str(out)  # internals never reach the response
    assert out["summary"]["flood"]["tiles_above_50pct"] >= 0 and out["summary"]["assessed"] == len(ok)


def test_result_is_valid_geojson_with_summary():
    out = area.assess_area(RECT, tile_m=1000, max_tiles=50)
    assert out["type"] == "FeatureCollection" and out["features"][0]["geometry"]["type"] == "Polygon"
    s = out["summary"]["flood"]
    assert s["max"] >= s["mean"] and "worst_tile" in s


client = TestClient(api.app)


def test_api_area_happy_path_and_validation():
    r = client.post("/assess/area", json={"geometry": RECT, "tile_m": 1000})
    assert r.status_code == 200 and r.json()["summary"]["assessed"] >= 4
    assert client.post("/assess/area", json={"geometry": {"type": "Point", "coordinates": [0, 0]}}).status_code == 422
    assert client.post("/assess/area", json={"geometry": RECT, "tile_m": 10}).status_code == 422


def test_api_area_is_rate_limited_by_tile_count(monkeypatch):
    monkeypatch.setenv("GEORISK_RATE_PER_MIN", "3")
    assert client.post("/assess/area", json={"geometry": RECT, "tile_m": 1000}).status_code == 429  # needs >3 tiles


def test_api_area_needs_key_when_configured(monkeypatch):
    monkeypatch.setenv("GEORISK_API_KEY", "k")
    assert client.post("/assess/area", json={"geometry": RECT}).status_code == 401
