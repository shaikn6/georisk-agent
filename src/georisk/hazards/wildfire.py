from __future__ import annotations

import numpy as np

from ..scoring import RiskScore
from .base import HazardContext, clip100, level


class WildfireHazard:
    name = "wildfire"
    uses_terrain = True

    def score(self, ctx: HazardContext) -> RiskScore:
        f, t = ctx.features, ctx.terrain
        # Fuel dryness only exists where there is fuel: measure NBR over vegetated pixels, not pavement.
        dryness = 0.0 if np.isnan(f.veg_mean_nbr) else max(0.0, (0.3 - f.veg_mean_nbr) / 0.5)
        dryness *= min(1.0, (f.dry_fuel_fraction + f.veg_fraction) / 0.25)  # negligible fuel -> negligible dryness signal
        # Dry fuel dominates, but dense live vegetation is still fuel load (it burns in drought).
        s = 110 * f.dry_fuel_fraction + 30 * dryness + 20 * f.built_fraction * f.dry_fuel_fraction + 18 * f.veg_fraction
        factors = [f"{f.dry_fuel_fraction:.0%} of the area is vegetated but dry (NDVI/NBR)"]
        if not np.isnan(f.veg_mean_nbr):
            factors.append(f"vegetation NBR {f.veg_mean_nbr:.2f} (lower = drier fuel)")
        if f.veg_fraction > 0.3:
            factors.append(f"{f.veg_fraction:.0%} of the area is dense vegetation (fuel load)")
        if t is not None and (f.dry_fuel_fraction > 0.05 or f.veg_fraction > 0.2):
            # Fire spreads faster uphill; slope only matters when there is fuel to burn.
            uplift = min(1.0, t.p90_slope_deg / 25.0)
            s *= 1.0 + 0.35 * uplift
            if t.p90_slope_deg > 10:
                factors.append(f"steep terrain (90th-percentile slope {t.p90_slope_deg:.0f} deg) accelerates spread")
        s = clip100(s)
        return RiskScore(self.name, s, level(s), factors)
