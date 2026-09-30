import numpy as np
import pytest

from georisk.scene import Scene
from georisk.scoring import extract_features, score_all
from georisk.synthetic import ARCHETYPES, make_scene


def scores(name):
    s = make_scene(name)
    return {r.name: r for r in score_all(extract_features(s))}


def test_archetype_ranks_the_expected_hazard_first():
    assert scores("floodplain")["flood"].score > scores("healthy_forest")["flood"].score
    assert scores("wildland_urban")["wildfire"].score > scores("healthy_forest")["wildfire"].score
    assert scores("dense_urban")["urban_density"].score > scores("healthy_forest")["urban_density"].score


def test_top_hazard_per_archetype():
    def top(n):
        return max(scores(n).values(), key=lambda r: r.score).name

    assert top("floodplain") == "flood"
    assert top("wildland_urban") == "wildfire"
    assert top("dense_urban") == "urban_density"


def test_scores_bounded_and_have_factors():
    for name in ARCHETYPES:
        for r in scores(name).values():
            assert 0 <= r.score <= 100 and r.factors and r.level in {"low", "moderate", "high", "severe"}


def test_cloud_masking_reduces_valid_fraction_not_scores_wildly():
    s = make_scene("wildland_urban")
    base = scores("wildland_urban")["wildfire"].score
    masked = Scene(bands=s.bands, valid=np.arange(s.valid.size).reshape(s.valid.shape) % 2 == 0, source="x")
    f = extract_features(masked)
    assert f.valid_fraction == pytest.approx(0.5, abs=0.01)
    assert abs(score_all(f)[1].score - base) < 10


def test_scene_rejects_missing_band():
    s = make_scene("floodplain")
    bad = dict(s.bands)
    bad.pop("nir")
    with pytest.raises(ValueError):
        Scene(bands=bad, valid=s.valid, source="x")


def test_synthetic_is_deterministic():
    a, b = make_scene("floodplain", seed=3), make_scene("floodplain", seed=3)
    assert np.array_equal(a.bands["nir"], b.bands["nir"])
