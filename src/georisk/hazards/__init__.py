"""Hazard registry and a runner that isolates failures per hazard."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..scoring import RiskScore
from .base import Hazard, HazardContext
from .flood import FloodHazard
from .urban import UrbanDensityHazard
from .wildfire import WildfireHazard

log = logging.getLogger(__name__)


@dataclass
class HazardOutcome:
    name: str
    status: str  # ok | degraded | failed
    score: RiskScore | None = None
    note: str | None = None


def default_hazards() -> list[Hazard]:
    return [FloodHazard(), WildfireHazard(), UrbanDensityHazard()]


def run_hazards(ctx: HazardContext, hazards: list[Hazard] | None = None) -> list[HazardOutcome]:
    """Score every hazard independently. One hazard raising never removes the others."""
    out: list[HazardOutcome] = []
    for h in hazards or default_hazards():
        try:
            s = h.score(ctx)
            degraded = h.uses_terrain and ctx.terrain is None
            out.append(HazardOutcome(h.name, "degraded" if degraded else "ok", s,
                                     "terrain unavailable; scored from imagery only" if degraded else None))
        except Exception as exc:
            log.exception("hazard %s failed", h.name)
            out.append(HazardOutcome(h.name, "failed", None, f"{type(exc).__name__}: {exc}"))
    return out


__all__ = ["Hazard", "HazardContext", "HazardOutcome", "default_hazards", "run_hazards"]
