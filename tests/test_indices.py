import numpy as np
import pytest

from georisk.indices import masked_mean, nbr, ndbi, ndvi, ndwi


def test_ndvi_known_values():
    nir, red = np.array([0.5]), np.array([0.1])
    assert ndvi(nir, red)[0] == pytest.approx((0.5 - 0.1) / 0.6, abs=1e-6)


def test_ndwi_positive_for_water():
    assert ndwi(np.array([0.08]), np.array([0.02]))[0] > 0.5


def test_ndbi_and_nbr_signs():
    assert ndbi(np.array([0.3]), np.array([0.2]))[0] > 0
    assert nbr(np.array([0.4]), np.array([0.1]))[0] > 0


def test_zero_division_is_safe():
    z = np.zeros((2, 2))
    assert np.isfinite(ndvi(z, z)).all()


def test_masked_mean_handles_empty_mask():
    v = np.ones((3, 3))
    assert np.isnan(masked_mean(v, np.zeros((3, 3), dtype=bool)))
    assert masked_mean(v, np.ones((3, 3), dtype=bool)) == 1.0
