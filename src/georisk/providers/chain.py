"""Failover orchestration: cache -> providers (each behind a breaker) -> stale cache -> synthetic."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..aoi import BBox
from ..config import Settings
from ..resilience import CircuitBreaker, CircuitOpen, retry
from ..scene import Scene
from .base import NoSceneFound, ProviderError, SceneProvider
from .cache import SceneCache

log = logging.getLogger(__name__)


class AllProvidersFailed(RuntimeError):
    def __init__(self, trail: list[dict]) -> None:
        super().__init__("all imagery sources failed: " + "; ".join(f"{t['source']}={t['outcome']}" for t in trail))
        self.trail = trail


@dataclass
class ChainResult:
    scene: Scene
    trail: list[dict] = field(default_factory=list)

    @property
    def degraded(self) -> bool:
        return any(t["outcome"] != "ok" for t in self.trail) or self.scene.meta.get("cache") == "stale"


class ProviderChain:
    def __init__(
        self,
        providers: list[SceneProvider],
        settings: Settings | None = None,
        cache: SceneCache | None = None,
        fallback: SceneProvider | None = None,
        sleep=None,
    ) -> None:
        if not providers and fallback is None:
            raise ValueError("at least one provider is required")
        self.providers = providers
        self.settings = settings or Settings()
        self.cache = cache
        self.fallback = fallback
        self.breakers = {p.name: CircuitBreaker(failure_threshold=2, cooldown=120.0) for p in providers}
        self._sleep = sleep

    def health(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for p in self.providers:
            try:
                p.probe()
                out[p.name] = {"reachable": True, "circuit": self.breakers[p.name].state}
            except ProviderError as exc:
                out[p.name] = {"reachable": False, "circuit": self.breakers[p.name].state, "error": str(exc)}
        return out

    def load(self, bbox: BBox, *, days: int | None = None, max_cloud: float | None = None, grid: int = 128) -> ChainResult:
        days = days or self.settings.lookback_days
        max_cloud = max_cloud if max_cloud is not None else self.settings.max_cloud
        trail: list[dict] = []
        key = self.cache.key(bbox, days, max_cloud, grid) if self.cache else None

        if self.cache and key:
            hit = self.cache.get(key)
            if hit:
                return ChainResult(hit, [{"source": "cache", "outcome": "ok"}])

        retry_kwargs: dict = {"attempts": self.settings.retry_attempts, "never_retry": (NoSceneFound,)}
        if self._sleep:
            retry_kwargs["sleep"] = self._sleep
        for p in self.providers:
            try:
                scene = self.breakers[p.name].call(
                    lambda p=p, kw=retry_kwargs: retry(
                        lambda: p.load(bbox, days=days, max_cloud=max_cloud, grid=grid), **kw
                    )
                )
                trail.append({"source": p.name, "outcome": "ok"})
                if self.cache and key:
                    self.cache.put(key, scene)
                return ChainResult(scene, trail)
            except CircuitOpen:
                trail.append({"source": p.name, "outcome": "circuit-open"})
            except NoSceneFound as exc:
                trail.append({"source": p.name, "outcome": f"no-scene: {exc}"})
            except Exception as exc:  # noqa: BLE001 - any provider failure means: try the next one
                trail.append({"source": p.name, "outcome": f"error: {exc}"})
                log.warning("provider %s failed: %s", p.name, exc)

        if self.cache and key:
            stale = self.cache.get(key, allow_stale=True)
            if stale:
                trail.append({"source": "cache-stale", "outcome": "ok"})
                return ChainResult(stale, trail)

        if self.fallback is not None:
            scene = self.fallback.load(bbox, days=days, max_cloud=max_cloud, grid=grid)
            trail.append({"source": self.fallback.name, "outcome": "ok"})
            return ChainResult(scene, trail)

        raise AllProvidersFailed(trail)


def build_default_chain(settings: Settings | None = None) -> ProviderChain:
    """Factory used by the service. Synthetic data is opt-in only (GEORISK_ALLOW_SYNTHETIC=1)."""
    from .earth_search import EarthSearchProvider
    from .planetary import PlanetaryComputerProvider
    from .synthetic import SyntheticProvider

    settings = settings or Settings()
    registry = {"earth-search": EarthSearchProvider, "planetary-computer": PlanetaryComputerProvider}
    providers = [registry[n]() for n in settings.providers if n in registry]
    return ProviderChain(
        providers,
        settings,
        cache=SceneCache(settings.cache_dir, settings.cache_ttl_s),
        fallback=SyntheticProvider() if settings.allow_synthetic_fallback else None,
    )
