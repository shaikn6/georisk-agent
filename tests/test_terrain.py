import numpy as np
import pytest

from georisk.aoi import bbox_from_point
from georisk.terrain import terrain_features

BBOX = bbox_from_point(40.0, -83.0, 500)  # ~1 km wide


def plane(slope_deg, n=64):
    width_m = 1000.0
    x = np.linspace(0, width_m, n)
    return np.tile(np.tan(np.radians(slope_deg)) * x, (n, 1))


def test_recovers_known_slope():
    t = terrain_features(plane(5.0), BBOX)
    assert t.mean_slope_deg == pytest.approx(5.0, abs=0.7)
    assert t.relief_m == pytest.approx(1000 * np.tan(np.radians(5.0)), rel=0.05)


def test_flat_ground_has_zero_slope_and_is_low_lying():
    t = terrain_features(np.full((32, 32), 200.0), BBOX)
    assert t.mean_slope_deg == pytest.approx(0.0, abs=1e-6)
    assert t.low_lying_fraction == 1.0


def test_center_percentile_tracks_position_in_a_bowl():
    y, x = np.mgrid[-1:1:64j, -1:1:64j]
    bowl = 10 * (x**2 + y**2)  # centre is the lowest point
    assert terrain_features(bowl, BBOX).center_percentile < 0.05
    assert terrain_features(-bowl, BBOX).center_percentile > 0.95  # hilltop


def test_nan_pixels_are_tolerated():
    d = plane(3.0)
    d[10:14, 10:14] = np.nan
    assert np.isfinite(terrain_features(d, BBOX).mean_slope_deg)


def test_rejects_degenerate_dem():
    with pytest.raises(ValueError):
        terrain_features(np.ones((2, 2)), BBOX)
    with pytest.raises(ValueError):
        terrain_features(np.full((10, 10), np.nan), BBOX)


def test_mosaic_takes_first_valid_pixel_and_never_invents_data():
    from georisk.terrain import combine_first_valid

    a = np.array([[1.0, np.nan], [np.nan, np.nan]])
    b = np.array([[9.0, 2.0], [np.nan, 4.0]])
    out = combine_first_valid([a, b])
    assert out[0, 0] == 1.0 and out[0, 1] == 2.0 and out[1, 1] == 4.0
    assert np.isnan(out[1, 0])  # nobody covered it: stays NaN, not 0
    with pytest.raises(ValueError):
        combine_first_valid([])


def test_single_pixel_spikes_do_not_create_steep_slopes():
    d = np.full((32, 32), 100.0)
    d[16, 16] = 400.0  # a "skyscraper"
    assert terrain_features(d, BBOX).p90_slope_deg < 1.0
