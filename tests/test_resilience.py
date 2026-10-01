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


class FakeClock:
    """Time advances only when a call 'takes time' or the retry loop sleeps."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def test_retry_gives_up_once_the_time_budget_is_spent_instead_of_retrying_a_hung_call():
    clock, calls = FakeClock(), []

    def hung():
        calls.append(clock.t)
        clock.t += 12.0  # each attempt hangs for 12 s before failing
        raise TimeoutError("read timed out")
    with pytest.raises(TimeoutError):
        retry(hung, attempts=5, base_delay=1.0, budget_s=20.0, sleep=clock.sleep, clock=clock)
    assert len(calls) == 2  # attempt 2 ends at ~25 s; without a budget it would have run all 5 (60+ s)


def test_retry_without_a_budget_keeps_its_old_behaviour_and_fast_failures_still_retry_within_budget():
    clock = FakeClock()
    n = {"i": 0}

    def flaky():
        n["i"] += 1
        if n["i"] < 3:
            raise OSError("blip")
        return "ok"
    assert retry(flaky, attempts=5, base_delay=0.5, budget_s=20.0, sleep=clock.sleep, clock=clock) == "ok" and n["i"] == 3
    n["i"] = 0
    assert retry(flaky, attempts=5, base_delay=0.5, sleep=clock.sleep, clock=clock) == "ok"


def test_the_chain_fails_over_after_one_hung_attempt_not_after_all_retries():
    from georisk.aoi import BBox
    from georisk.config import Settings
    from georisk.providers.base import ProviderError
    from georisk.providers.chain import ProviderChain
    from georisk.scene import BANDS, Scene

    clock = FakeClock()
    seen = []

    class Hung:
        name = "slow"

        def load(self, bbox, *, days, max_cloud, grid):
            seen.append("slow")
            clock.t += 25.0
            raise ProviderError("timed out")

    class Good:
        name = "fast"

        def load(self, bbox, *, days, max_cloud, grid):
            seen.append("fast")
            import numpy as np

            return Scene(bands={b: np.full((4, 4), 0.2) for b in BANDS}, valid=np.ones((4, 4), bool), source="fast:x")
    import dataclasses

    s = dataclasses.replace(Settings(), retry_attempts=4, provider_budget_s=20.0)
    chain = ProviderChain([Hung(), Good()], settings=s, sleep=clock.sleep, clock=clock)
    res = chain.load(BBox(-83.02, 39.95, -83.0, 39.97))
    assert seen == ["slow", "fast"] and res.scene.source == "fast:x"  # one slow attempt, then straight to the next provider
