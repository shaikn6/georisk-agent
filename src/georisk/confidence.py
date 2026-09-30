"""How much should a consumer trust this assessment? Transparent, additive penalties."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from .scene import Scene


@dataclass
class Confidence:
    score: float
    level: str
    reasons: list[str]


def _age_days(scene: Scene, now: datetime) -> float | None:
    if not scene.acquired:
        return None
    return (now - datetime.fromisoformat(scene.acquired)).total_seconds() / 86400


def assess_confidence(
    scene: Scene, *, terrain_ok: bool, degraded_path: bool, now: datetime | None = None
) -> Confidence:
    now = now or datetime.now(timezone.utc)
    if scene.source.startswith("synthetic"):
        return Confidence(0.0, "none", ["synthetic data: not a real observation"])
    score, reasons = 1.0, []
    if scene.valid_fraction < 0.95:
        score -= (1.0 - scene.valid_fraction) * 0.6
        reasons.append(f"only {scene.valid_fraction:.0%} of pixels usable (clouds/shadow masked)")
    if scene.cloud_cover and scene.cloud_cover > 10:
        score -= min(0.2, scene.cloud_cover / 200)
        reasons.append(f"scene cloud cover {scene.cloud_cover:.0f}%")
    age = _age_days(scene, now)
    if age is not None and age > 30:
        score -= 0.15 if age <= 60 else 0.3
        reasons.append(f"imagery is {age:.0f} days old")
    if not terrain_ok:
        score -= 0.1
        reasons.append("no terrain data; flood/fire scores use imagery only")
    if degraded_path:
        score -= 0.05
        reasons.append("primary imagery source unavailable; used a fallback")
    if scene.meta.get("cache") == "stale":
        score -= 0.15
        reasons.append("served from an expired cache entry")
    score = round(max(0.0, min(1.0, score)), 2)
    return Confidence(score, "high" if score >= 0.8 else "medium" if score >= 0.55 else "low", reasons or ["good coverage, recent, clear"])
