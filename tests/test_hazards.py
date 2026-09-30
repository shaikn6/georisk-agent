import numpy as np
import pytest

from georisk.hazards import HazardContext, run_hazards
from georisk.hazards.flood import FloodHazard
from georisk.hazards.wildfire import WildfireHazard
from georisk.scene import Scene
from georisk.scoring import extract_features
from georisk.synthetic import ARCHETYPES, make_scene
from georisk.terrain import TerrainFeatures


def ctx(archetype, terrain=None):
    return HazardContext(extract_features(make_scene(archetype)), terrain)


LOW_FLAT = TerrainFeatures(0.5, 1.0, 3.0, 0.05, 0.9)
HIGH_STEEP = TerrainFeatures(20.0, 30.0, 300.0, 0.9, 0.0)


def by_name(outcomes):
    return {o.name: o for o in outcomes}


def test_each_archetype_ranks_its_own_hazard_first():
    def top(a):
        oc = [o for o in run_hazards(ctx(a)) if o.score]
        return max(oc, key=lambda o: o.score.score).name

    assert top("floodplain") == "flood"
    assert top("wildland_urban") == "wildfire"
    assert top("dense_urban") == "urban_density"


def test_failed_hazard_is_isolated_others_still_score():
    class Broken:
        name = "broken"
        uses_terrain = False

        def score(self, c):
            raise RuntimeError("model file missing")

    out = by_name(run_hazards(ctx("floodplain"), [FloodHazard(), Broken(), WildfireHazard()]))
    assert out["broken"].status == "failed" and "model file missing" in out["broken"].note
    assert out["flood"].score is not None and out["wildfire"].score is not None


def test_missing_terrain_marks_terrain_hazards_degraded_not_failed():
    out = by_name(run_hazards(ctx("wildland_urban")))
    assert out["flood"].status == "degraded" and out["wildfire"].status == "degraded"
    assert out["urban_density"].status == "ok"
    assert out["flood"].score is not None


def test_low_flat_ground_raises_flood_score():
    base = FloodHazard().score(ctx("floodplain")).score
    assert FloodHazard().score(ctx("floodplain", LOW_FLAT)).score > base
    assert FloodHazard().score(ctx("floodplain", HIGH_STEEP)).score <= base + 1


def test_steep_slope_raises_wildfire_score_only_when_fuel_exists():
    base = WildfireHazard().score(ctx("wildland_urban")).score
    assert WildfireHazard().score(ctx("wildland_urban", HIGH_STEEP)).score > base
    # Dense live vegetation on a steep slope is fuel too: slope must raise it, but it must
    # still score below genuinely dry fuel on the same slope.
    forest_flat = WildfireHazard().score(ctx("healthy_forest")).score
    forest_steep = WildfireHazard().score(ctx("healthy_forest", HIGH_STEEP)).score
    assert forest_steep > forest_flat
    assert forest_steep < WildfireHazard().score(ctx("wildland_urban", HIGH_STEEP)).score
    bare = WildfireHazard().score(ctx("dense_urban", HIGH_STEEP)).score
    assert bare < forest_steep  # little vegetation: little to burn even on a slope


def test_scores_bounded_and_explained_for_every_archetype_and_terrain():
    for a in ARCHETYPES:
        for t in (None, LOW_FLAT, HIGH_STEEP):
            for o in run_hazards(ctx(a, t)):
                assert 0 <= o.score.score <= 100 and o.score.factors


def test_cloud_masking_keeps_scores_stable():
    s = make_scene("wildland_urban")
    base = WildfireHazard().score(HazardContext(extract_features(s))).score
    half = Scene(bands=s.bands, valid=np.arange(s.valid.size).reshape(s.valid.shape) % 2 == 0, source="x")
    assert abs(WildfireHazard().score(HazardContext(extract_features(half))).score - base) < 10


def test_synthetic_is_deterministic():
    assert np.array_equal(make_scene("floodplain", seed=3).bands["nir"], make_scene("floodplain", seed=3).bands["nir"])


def test_scene_rejects_missing_band():
    s = make_scene("floodplain")
    bad = dict(s.bands)
    bad.pop("nir")
    with pytest.raises(ValueError):
        Scene(bands=bad, valid=s.valid, source="x")


def test_built_surfaces_are_not_counted_as_dry_vegetation():
    """Regression: urban pixels (NDVI ~0.1) were classed as 'dry fuel', giving cities max wildfire risk."""
    f = extract_features(make_scene("dense_urban"))
    assert f.dry_fuel_fraction < 0.1 and f.built_fraction > 0.7
    assert WildfireHazard().score(HazardContext(f, HIGH_STEEP)).score < 40
