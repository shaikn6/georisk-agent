"""Hazard plug-in contract. Each hazard is independent: it can fail without affecting others."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..scoring import Features, RiskScore
from ..terrain import TerrainFeatures


@dataclass(frozen=True)
class HazardContext:
    features: Features
    terrain: TerrainFeatures | None = None


class Hazard(Protocol):
    name: str
    uses_terrain: bool

    def score(self, ctx: HazardContext) -> RiskScore: ...


def clip100(x: float) -> float:
    return float(min(100.0, max(0.0, x)))


def level(score: float) -> str:
    return "low" if score < 25 else "moderate" if score < 50 else "high" if score < 75 else "severe"
