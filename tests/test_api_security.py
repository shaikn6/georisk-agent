import pytest
from fastapi.testclient import TestClient

from georisk import api

client = TestClient(api.app)
OK = {"lat": 39.96, "lon": -83.0, "archetype": "floodplain", "use_llm": False}


def test_open_when_no_key_configured(monkeypatch):
    monkeypatch.delenv("GEORISK_API_KEY", raising=False)
    assert client.post("/assess", json=OK).status_code == 200


def test_key_required_when_configured(monkeypatch):
    monkeypatch.setenv("GEORISK_API_KEY", "s3cret")
    assert client.post("/assess", json=OK).status_code == 401
    assert client.post("/assess", json=OK, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/assess", json=OK, headers={"X-API-Key": "s3cret"}).status_code == 200
    assert client.get("/health").status_code == 200  # liveness stays open


@pytest.mark.parametrize("bad", [
    {"lat": 91, "lon": 0}, {"lat": 0, "lon": 181}, {"lat": "x", "lon": 0}, {"lat": 0, "lon": 0, "radius_m": 1e9},
    {"lat": 0, "lon": 0, "composite": 99}, {"lat": 0, "lon": 0, "days": 0}, {},
])
def test_malformed_input_rejected_with_422(bad):
    assert client.post("/assess", json=bad).status_code == 422


def test_nan_and_inf_rejected():
    r = client.post("/assess", content='{"lat": NaN, "lon": 0}', headers={"content-type": "application/json"})
    assert r.status_code == 422


def test_unknown_archetype_is_422_not_500():
    assert client.post("/assess", json={**OK, "archetype": "../../etc/passwd"}).status_code == 422


def test_batch_errors_do_not_leak_internals(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("secret path /Users/x/.aws/credentials")
    monkeypatch.setattr(api, "assess", boom)
    body = client.post("/assess/batch", json={"items": [OK]}).json()
    assert body["results"][0] == {"ok": False, "error": "internal error"}
    assert "secret" not in str(body)


def test_batch_size_capped():
    assert client.post("/assess/batch", json={"items": [OK] * 26}).status_code == 422


def test_overload_returns_429_and_releases_slots(monkeypatch):
    sem = api.threading.BoundedSemaphore(1)
    monkeypatch.setattr(api, "_SLOTS", sem)
    sem.acquire()
    assert client.post("/assess", json=OK).status_code == 429
    sem.release()
    assert client.post("/assess", json=OK).status_code == 200
    assert client.post("/assess", json=OK).status_code == 200  # slot was released after success


def test_llm_ignored_unless_explicitly_allowed(monkeypatch):
    seen = {}
    monkeypatch.delenv("GEORISK_ALLOW_LLM", raising=False)
    real = api.assess
    monkeypatch.setattr(api, "assess", lambda *a, **k: (seen.update(k), real(*a, **k))[1])
    client.post("/assess", json={**OK, "use_llm": True})
    assert seen["use_llm"] is False
