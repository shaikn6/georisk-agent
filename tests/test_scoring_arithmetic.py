"""Pin the feature-extraction and hazard-scoring arithmetic to hand-computed values.

The other hazard tests work through the synthetic archetypes and assert relative behaviour
(this ranks higher than that). These assert the actual numbers, so a changed threshold or
coefficient breaks a test instead of silently shifting every score.
"""

import numpy as np
import pytest

from georisk.hazards import HazardContext
from georisk.hazards.base import clip100, level
from georisk.hazards.flood import FloodHazard
from georisk.hazards.wildfire import WildfireHazard
from georisk.scene import Scene
from georisk.scoring import Features, extract_features
from georisk.terrain import TerrainFeatures

# Four pixels, one per class, chosen so every index lands clearly inside (never on) its cutoff.
# A: open water   B: built surface   C: dry fuel (vegetated, NDVI 0.2-0.3)   D: dense green vegetation
_BANDS = {
    "green": np.array([0.30, 0.10, 0.10, 0.10]),
    "red":   np.array([0.10, 0.10, 0.15, 0.20]),
    "nir":   np.array([0.10, 0.10, 0.25, 0.80]),
    "swir1": np.array([0.10, 0.30, 0.10, 0.10]),
    "swir2": np.array([0.10, 0.10, 0.25, 0.20]),
}


def _scene(valid):
    return Scene(bands={k: v.copy() for k, v in _BANDS.items()}, valid=np.array(valid), source="test")


def test_extract_features_counts_classes_on_a_hand_built_scene():
    f = extract_features(_scene([True, True, True, True]))
    assert f.water_fraction == pytest.approx(0.25)      # only pixel A reads as water (NDWI 0.5)
    assert f.built_fraction == pytest.approx(0.25)      # only pixel B (NDBI 0.5, NDVI 0)
    assert f.dry_fuel_fraction == pytest.approx(0.25)   # only pixel C (NDVI 0.25 in [0.2,0.5), NBR 0)
    assert f.veg_fraction == pytest.approx(0.25)        # only pixel D (NDVI 0.6 > 0.3)
    assert f.valid_fraction == pytest.approx(1.0)


def test_extract_features_veg_mean_nbr_uses_only_vegetated_pixels():
    f = extract_features(_scene([True, True, True, True]))
    # Vegetated pixels (NDVI >= 0.2) are C (NBR 0) and D (NBR 0.6); their mean is 0.3.
    assert f.veg_mean_nbr == pytest.approx(0.30, abs=1e-4)
    # The all-pixel mean NBR is diluted by the two bare pixels: (0 + 0 + 0 + 0.6) / 4.
    assert f.mean_nbr == pytest.approx(0.15, abs=1e-4)
    assert f.veg_mean_nbr != pytest.approx(f.mean_nbr)


def test_extract_features_uses_the_valid_pixel_denominator():
    # Keep only pixel A (water) and pixel D (dense veg) valid.
    f = extract_features(_scene([True, False, False, True]))
    assert f.valid_fraction == pytest.approx(0.5)
    assert f.water_fraction == pytest.approx(0.5)   # 1 water of 2 valid
    assert f.veg_fraction == pytest.approx(0.5)     # 1 dense-veg of 2 valid
    assert f.built_fraction == pytest.approx(0.0)   # pixel B is masked out
    assert f.dry_fuel_fraction == pytest.approx(0.0)


def _features(**kw):
    base = {"water_fraction": 0.0, "mean_ndvi": 0.0, "mean_nbr": 0.0, "dry_fuel_fraction": 0.0,
            "built_fraction": 0.0, "valid_fraction": 1.0, "veg_fraction": 0.0, "veg_mean_nbr": float("nan")}
    base.update(kw)
    return Features(**base)


def test_flood_score_is_water_plus_built_when_no_terrain():
    s = FloodHazard().score(HazardContext(_features(water_fraction=0.5, built_fraction=0.2)))
    assert s.score == pytest.approx(140 * 0.5 + 35 * 0.2)  # 77.0
    assert s.level == "severe"


def test_flood_lowland_term_adds_on_flat_ground_and_vanishes_on_slope():
    f = _features(water_fraction=0.1)                       # base = 14.0
    flat = TerrainFeatures(0.0, 0.0, 0.0, 0.0, 0.0)         # lowest, flat -> full lowland term
    steep = TerrainFeatures(6.0, 0.0, 0.0, 0.0, 0.0)        # mean slope 6 deg zeroes the term
    assert FloodHazard().score(HazardContext(f, flat)).score == pytest.approx(14.0 + 25.0)
    assert FloodHazard().score(HazardContext(f, steep)).score == pytest.approx(14.0)


def test_wildfire_nan_dryness_is_safe_and_finite():
    # No vegetation NBR -> the dryness term drops out entirely, leaving 110 * dry_fuel.
    s = WildfireHazard().score(HazardContext(_features(dry_fuel_fraction=0.2)))
    assert s.score == pytest.approx(110 * 0.2)  # 22.0
    assert s.level == "low"


def test_wildfire_slope_uplift_is_capped_at_35_percent():
    f = _features(dry_fuel_fraction=0.1)                    # base = 11.0, dryness term 0 (NBR is NaN)
    flat = TerrainFeatures(0.0, 0.0, 0.0, 0.0, 0.0)
    very_steep = TerrainFeatures(0.0, 1000.0, 0.0, 0.0, 0.0)  # p90 slope saturates the uplift
    flat_score = WildfireHazard().score(HazardContext(f, flat)).score
    steep_score = WildfireHazard().score(HazardContext(f, very_steep)).score
    assert flat_score == pytest.approx(11.0)
    assert steep_score == pytest.approx(11.0 * 1.35)        # uplift never exceeds +35%


def test_level_and_clip_boundaries():
    assert level(24.9) == "low"
    assert level(25.0) == "moderate"
    assert level(49.9) == "moderate"
    assert level(50.0) == "high"
    assert level(74.9) == "high"
    assert level(75.0) == "severe"
    assert clip100(-5.0) == 0.0
    assert clip100(150.0) == 100.0
    assert clip100(50.0) == 50.0


def test_scene_rejects_a_band_whose_shape_differs_from_the_mask():
    bands = {k: v.copy() for k, v in _BANDS.items()}
    bands["red"] = np.zeros(3)  # wrong length
    with pytest.raises(ValueError):
        Scene(bands=bands, valid=np.ones(4, dtype=bool), source="test")
