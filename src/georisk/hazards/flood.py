from __future__ import annotations

from ..scoring import RiskScore
from .base import HazardContext, clip100, level


class FloodHazard:
    name = "flood"
    uses_terrain = True

    def score(self, ctx: HazardContext) -> RiskScore:
        f, t = ctx.features, ctx.terrain
        s = 140 * f.water_fraction + 35 * f.built_fraction
        factors = [f"{f.water_fraction:.0%} of the area reads as open water (NDWI)"]
        if f.built_fraction > 0.3:
            factors.append(f"{f.built_fraction:.0%} impervious surface limits drainage")
        if t is not None:
            # Low, flat ground collects water; high or steep ground sheds it.
            lowland = (1.0 - t.center_percentile) * max(0.0, 1.0 - t.mean_slope_deg / 6.0)
            s += 25 * lowland
            if t.center_percentile < 0.25 and t.mean_slope_deg < 3:
                factors.append(
                    f"site is in the lowest quarter of a flat AOI (mean slope {t.mean_slope_deg:.1f} deg)"
                )
            if t.low_lying_fraction > 0.4:
                factors.append(f"{t.low_lying_fraction:.0%} of the AOI is within 2 m of its lowest point")
        s = clip100(s)
        return RiskScore(self.name, s, level(s), factors)
