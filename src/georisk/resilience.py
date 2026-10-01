"""Failure-handling primitives: retry with backoff and a circuit breaker.

Kept dependency-free and clock-injectable so they are trivially unit-testable.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar

T = TypeVar("T")
log = logging.getLogger(__name__)


class CircuitOpen(RuntimeError):
    """Raised instead of calling a dependency that is currently failing fast."""


def retry(
    fn: Callable[[], T],
    *,
    attempts: int = 3,
    base_delay: float = 0.5,
    factor: float = 2.0,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
    never_retry: tuple[type[BaseException], ...] = (),
    sleep: Callable[[float], None] = time.sleep,
    budget_s: float | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> T:
    """Call ``fn`` up to ``attempts`` times with exponential backoff.

    ``budget_s`` bounds the total time spent on this call: once the time used plus the next backoff would exceed it, the last
    error is raised instead of retrying, so a hung provider cannot eat the whole request (callers then fail over).
    """
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    delay = base_delay
    t0 = clock()
    for i in range(1, attempts + 1):
        try:
            return fn()
        except never_retry:
            raise
        except retry_on as exc:
            if i == attempts or (budget_s is not None and clock() - t0 + delay >= budget_s):
                raise
            log.warning("attempt %d/%d failed (%s); retrying in %.1fs", i, attempts, exc, delay)
            sleep(delay)
            delay *= factor
    raise AssertionError("unreachable")  # pragma: no cover


@dataclass
class CircuitBreaker:
    """Opens after ``failure_threshold`` consecutive failures; half-opens after ``cooldown`` s."""

    failure_threshold: int = 3
    cooldown: float = 60.0
    clock: Callable[[], float] = time.monotonic
    _failures: int = field(default=0, init=False)
    _opened_at: float | None = field(default=None, init=False)

    @property
    def state(self) -> str:
        if self._opened_at is None:
            return "closed"
        return "half-open" if self.clock() - self._opened_at >= self.cooldown else "open"

    def call(self, fn: Callable[[], T]) -> T:
        if self.state == "open":
            raise CircuitOpen("circuit open; skipping call")
        try:
            result = fn()
        except Exception:
            self._failures += 1
            if self._failures >= self.failure_threshold or self.state == "half-open":
                self._opened_at = self.clock()
            raise
        self._failures = 0
        self._opened_at = None
        return result
