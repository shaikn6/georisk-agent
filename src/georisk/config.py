"""Environment-driven configuration (no hidden globals)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _f(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


@dataclass(frozen=True)
class Settings:
    cache_dir: Path = field(default_factory=lambda: Path(os.environ.get("GEORISK_CACHE_DIR", ".georisk_cache")))
    cache_ttl_s: float = field(default_factory=lambda: _f("GEORISK_CACHE_TTL_S", 6 * 3600))
    max_cloud: float = field(default_factory=lambda: _f("GEORISK_MAX_CLOUD", 30.0))
    lookback_days: int = field(default_factory=lambda: int(_f("GEORISK_LOOKBACK_DAYS", 60)))
    http_timeout_s: float = field(default_factory=lambda: _f("GEORISK_HTTP_TIMEOUT_S", 30.0))
    retry_attempts: int = field(default_factory=lambda: int(_f("GEORISK_RETRY_ATTEMPTS", 3)))
    providers: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            p.strip() for p in os.environ.get("GEORISK_PROVIDERS", "earth-search,planetary-computer").split(",") if p.strip()
        )
    )
    allow_synthetic_fallback: bool = field(
        default_factory=lambda: os.environ.get("GEORISK_ALLOW_SYNTHETIC", "0") == "1"
    )
