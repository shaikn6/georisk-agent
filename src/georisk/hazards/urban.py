from __future__ import annotations

from ..scoring import RiskScore
from .base import HazardContext, clip100, level


class UrbanDensityHazard:
    name = "urban_density"
    uses_terrain = False

    def score(self, ctx: HazardContext) -> RiskScore:
        s = clip100(115 * ctx.features.built_fraction)
        return RiskScore(self.name, s, level(s), [f"{ctx.features.built_fraction:.0%} of the area is built-up (NDBI/NDVI)"])
