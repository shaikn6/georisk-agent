import pytest

from georisk.resilience import CircuitBreaker, CircuitOpen, retry


def test_retry_succeeds_after_transient_failures():
    calls, sleeps = [], []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise OSError("boom")
        return "ok"

    assert retry(flaky, attempts=3, base_delay=1, sleep=sleeps.append) == "ok"
    assert sleeps == [1, 2]  # exponential backoff


def test_retry_gives_up_and_reraises():
    with pytest.raises(OSError):
        retry(lambda: (_ for _ in ()).throw(OSError("x")), attempts=2, sleep=lambda s: None)


def test_never_retry_short_circuits():
    calls = []

    def f():
        calls.append(1)
        raise KeyError("permanent")

    with pytest.raises(KeyError):
        retry(f, attempts=5, never_retry=(KeyError,), sleep=lambda s: None)
    assert len(calls) == 1


def test_circuit_opens_then_half_opens_then_closes():
    t = {"now": 0.0}
    cb = CircuitBreaker(failure_threshold=2, cooldown=10, clock=lambda: t["now"])

    def bad():
        raise RuntimeError("down")

    for _ in range(2):
        with pytest.raises(RuntimeError):
            cb.call(bad)
    assert cb.state == "open"
    with pytest.raises(CircuitOpen):
        cb.call(lambda: 1)  # fails fast, dependency not touched
    t["now"] = 11
    assert cb.state == "half-open"
    assert cb.call(lambda: 42) == 42
    assert cb.state == "closed"


def test_half_open_failure_reopens_immediately():
    t = {"now": 0.0}
    cb = CircuitBreaker(failure_threshold=1, cooldown=5, clock=lambda: t["now"])
    with pytest.raises(RuntimeError):
        cb.call(lambda: (_ for _ in ()).throw(RuntimeError()))
    t["now"] = 6
    with pytest.raises(RuntimeError):
        cb.call(lambda: (_ for _ in ()).throw(RuntimeError()))
    assert cb.state == "open"
