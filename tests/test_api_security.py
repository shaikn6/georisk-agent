import pytest
from fastapi.testclient import TestClient

from georisk import api

client = TestClient(api.app)


@pytest.fixture(autouse=True)
def _fresh_rate_limiter(monkeypatch):
    """Every test gets an empty limiter and a generous limit unless it sets its own."""
    api._HITS.clear()
    monkeypatch.setenv("GEORISK_RATE_PER_MIN", "1000")
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


def test_rate_limit_is_per_client_and_recovers(monkeypatch):
    monkeypatch.setenv("GEORISK_RATE_PER_MIN", "3")
    assert all(api.take("1.1.1.1", now=100.0) for _ in range(3))
    assert not api.take("1.1.1.1", now=101.0)  # 4th in the window
    assert api.take("2.2.2.2", now=101.0)  # other clients unaffected
    assert api.take("1.1.1.1", now=161.0)  # window slid


def test_batch_cost_counts_every_item(monkeypatch):
    monkeypatch.setenv("GEORISK_RATE_PER_MIN", "5")
    assert not api.take("9.9.9.9", cost=6, now=1.0)
    assert api.take("9.9.9.9", cost=5, now=1.0)


def test_forwarded_header_ignored_unless_proxy_trusted(monkeypatch):
    req = type("R", (), {"headers": {"x-forwarded-for": "6.6.6.6, 10.0.0.1"},
                         "client": type("C", (), {"host": "127.0.0.1"})()})()
    monkeypatch.delenv("GEORISK_TRUST_PROXY", raising=False)
    assert api.client_ip(req) == "127.0.0.1"  # spoofable header ignored by default
    monkeypatch.setenv("GEORISK_TRUST_PROXY", "1")
    assert api.client_ip(req) == "6.6.6.6"


def test_security_headers_and_no_store():
    r = client.post("/assess", json=OK)
    assert r.headers["x-frame-options"] == "DENY" and "max-age" in r.headers["strict-transport-security"]
    assert r.headers["cache-control"] == "no-store"


def test_request_id_is_echoed_and_generated():
    assert client.get("/health", headers={"x-request-id": "abc123"}).headers["x-request-id"] == "abc123"
    assert len(client.get("/health").headers["x-request-id"]) == 16
    assert len(client.get("/health", headers={"x-request-id": "a" * 500}).headers["x-request-id"]) == 64  # capped


def test_metrics_expose_bounded_labels_and_no_coordinates():
    client.post("/assess", json=OK)
    client.get("/nope-" + "x" * 50)
    text = client.get("/metrics").text
    assert 'route="/assess",status_class="2xx"' in text and "georisk_request_duration_seconds_count" in text
    assert "nope-xxxx" not in text and "39.96" not in text  # raw URLs and coordinates never become labels


def test_metrics_respect_api_key(monkeypatch):
    monkeypatch.setenv("GEORISK_API_KEY", "k")
    assert client.get("/metrics").status_code == 401
    assert client.get("/metrics", headers={"X-API-Key": "k"}).status_code == 200


def test_structured_log_has_no_location(caplog):
    import json as _json

    with caplog.at_level("INFO", logger="georisk.api"):
        client.post("/assess", json=OK)
    events = [_json.loads(r.message) for r in caplog.records if r.message.startswith("{")]
    assert {"http_request", "assessment"} <= {e["event"] for e in events}
    assert "39.96" not in caplog.text and "-83.0" not in caplog.text
