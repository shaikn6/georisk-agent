"""Arithmetic for the calibrated-probability layer: the median fill, the track-record lookup,
the leave-region-out skip, and the shipped artifact's internal consistency."""

import numpy as np
import pytest

from georisk.calibration import _vector, load_artifact, similar_sites
from georisk.scoring import Features
from georisk.terrain import TerrainFeatures
from georisk.validation.metrics import leave_group_out

_T = TerrainFeatures(mean_slope_deg=1.0, p90_slope_deg=2.0, relief_m=9.0,
                     center_percentile=0.1, low_lying_fraction=0.2)


def _features(**kw):
    base = {"water_fraction": 0.0, "mean_ndvi": 0.0, "mean_nbr": 0.0, "dry_fuel_fraction": 0.0,
            "built_fraction": 0.0, "valid_fraction": 1.0, "veg_fraction": 0.0, "veg_mean_nbr": float("nan")}
    base.update(kw)
    return Features(**base)


def test_vector_fills_nan_and_absent_inputs_with_the_spec_median():
    # water_fraction is NaN -> fill; t_relief_m is present -> real value kept.
    spec = {"features": ["water_fraction", "t_relief_m"], "fill": {"water_fraction": 0.2, "t_relief_m": 5.0}}
    row = _vector(spec, _features(water_fraction=float("nan")), _T)
    assert row.tolist() == [[0.2, 9.0]]

    # A grid feature with grid data absent is also filled rather than guessed elsewhere.
    grid_spec = {"features": ["g_dist_km"], "fill": {"g_dist_km": 3.0}}
    assert _vector(grid_spec, _features(), _T, grid=None).tolist() == [[3.0]]


def test_similar_sites_matches_the_band_handles_p_equals_one_and_misses_cleanly():
    spec = {"reliability": [
        {"lo": 0.2, "hi": 0.5, "n": 10, "rate": 0.2, "ci": [0.1, 0.3]},
        {"lo": 0.5, "hi": 1.0001, "n": 20, "rate": 0.8, "ci": [0.7, 0.9]},
    ]}
    mid = similar_sites(spec, 0.3)
    assert mid == {"n": 10, "observed_pct": 20, "ci95_pct": [10, 30], "band_pct": [20, 50]}
    top = similar_sites(spec, 1.0)   # the top band's hi >= 1.0 catches a probability of exactly 1
    assert top["observed_pct"] == 80 and top["n"] == 20
    assert similar_sites(spec, 0.1) is None   # below every band -> no evidence rather than a wrong band


def test_leave_group_out_leaves_nan_when_a_held_out_region_strands_a_single_class():
    rng = np.random.default_rng(0)
    # Region A is mixed; B and C are all positive. Holding A out leaves B+C, which is single-class,
    # so A cannot be scored and its out-of-fold predictions stay NaN.
    groups = np.array(["A"] * 6 + ["B"] * 4 + ["C"] * 4)
    y = np.array([0, 0, 0, 1, 1, 1] + [1] * 4 + [1] * 4)
    X = rng.normal(size=(len(y), 2))
    oof = leave_group_out(X, y, groups)["oof"]
    assert np.isnan(oof[groups == "A"]).all()
    assert not np.isnan(oof[groups == "B"]).any()
    assert not np.isnan(oof[groups == "C"]).any()


@pytest.mark.skipif(not load_artifact(), reason="no shipped calibration artifact")
def test_shipped_artifact_schema_is_internally_consistent():
    for task, spec in load_artifact().items():
        assert all(c in spec["fill"] for c in spec["features"]), f"{task}: fill misses a feature"
        bands = spec.get("reliability", [])
        for lo, hi in zip([b["hi"] for b in bands], [b["lo"] for b in bands[1:]]):
            assert lo == hi, f"{task}: reliability bands overlap or leave a gap"
        for b in bands:
            assert 0.0 <= b["lo"] < b["hi"] <= 1.0001 and b["n"] > 0, f"{task}: bad band {b}"
        dom = spec.get("domain")
        if dom is not None:
            assert len(dom) == 4 and dom[0] <= dom[1] and dom[2] <= dom[3], f"{task}: bad domain {dom}"
