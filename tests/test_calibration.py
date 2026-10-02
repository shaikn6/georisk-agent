import numpy as np

from georisk.calibration import calibrated_probabilities, load_artifact
from georisk.scoring import extract_features
from georisk.synthetic import make_scene
from georisk.terrain import TerrainFeatures
from georisk.validation.ensemble import load_model
from georisk.validation.metrics import Logistic

T = TerrainFeatures(1.0, 2.0, 5.0, 0.1, 0.8)


def fake_artifact(weight_on_water=3.0):
    cols = ["water_fraction", "t_relief_m"]
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.random(200), rng.random(200) * 10])
    y = (rng.random(200) < 1 / (1 + np.exp(-weight_on_water * (X[:, 0] - 0.5) * 4))).astype(int)
    return {"flood": {"features": cols, "fill": {"water_fraction": 0.2, "t_relief_m": 5.0},
                      "model": Logistic(1.0).fit(X, y).to_dict(), "meaning": "test"}}


def test_probability_is_valid_and_responds_to_the_signal():
    art = fake_artifact()
    wet = extract_features(make_scene("floodplain"))
    dry = extract_features(make_scene("healthy_forest"))
    p_wet = calibrated_probabilities(wet, T, art)["flood"]["probability"]
    p_dry = calibrated_probabilities(dry, T, art)["flood"]["probability"]
    assert 0 <= p_dry < p_wet <= 1


def test_no_terrain_or_no_model_means_no_probability_not_an_error():
    f = extract_features(make_scene("floodplain"))
    assert calibrated_probabilities(f, None, fake_artifact()) == {}
    assert calibrated_probabilities(f, T, {}) == {}


def test_broken_artifact_entry_is_ignored():
    f = extract_features(make_scene("floodplain"))
    bad = {"flood": {"features": ["nope"], "fill": {}, "model": {}, "meaning": "x"}}
    assert calibrated_probabilities(f, T, bad) == {}


def test_shipped_artifact_loads_and_is_well_formed_if_present():
    art = load_artifact()
    for spec in art.values():
        assert set(spec) >= {"features", "fill", "model", "meaning"}
        model = load_model(spec["model"])  # whichever family was served must deserialise and predict
        row = np.array([[spec["fill"][c] for c in spec["features"]]])
        assert 0.0 <= float(model.predict_proba(row)[0]) <= 1.0


def test_ensemble_members_learn_and_roundtrip_through_json():
    import json

    from georisk.validation.ensemble import Ensemble, load_model

    rng = np.random.default_rng(1)
    X = rng.normal(size=(150, 3))
    y = ((X[:, 0] + 0.5 * X[:, 1] ** 2 + rng.normal(scale=0.3, size=150)) > 0.4).astype(int)
    ens = Ensemble().fit(X, y)
    for name, p in ens.member_probas(X).items():
        assert ((p >= 0) & (p <= 1)).all(), name
    back = load_model(json.loads(json.dumps(ens.to_dict())))
    assert np.allclose(back.predict_proba(X), ens.predict_proba(X), atol=1e-3)
    from georisk.validation.metrics import auc

    assert auc(y, ens.predict_proba(X)) > 0.8


def test_legacy_logistic_artifact_still_loads():
    from georisk.validation.ensemble import load_model
    from georisk.validation.metrics import Logistic

    X = np.random.default_rng(2).normal(size=(50, 2))
    y = (X[:, 0] > 0).astype(int)
    m = Logistic().fit(X, y)
    assert np.allclose(load_model(m.to_dict()).predict_proba(X), m.predict_proba(X))


def test_probability_withheld_outside_training_domain():
    art = fake_artifact()
    art["flood"]["domain"] = [30.0, 40.0, -100.0, -80.0]
    f = extract_features(make_scene("floodplain"))
    assert "flood" in calibrated_probabilities(f, T, art, lat=35.0, lon=-90.0)
    assert calibrated_probabilities(f, T, art, lat=48.0, lon=-90.0) == {}  # outside the box: no extrapolation


def test_sklearn_forest_exports_to_numpy_and_matches():
    import pytest

    pytest.importorskip("sklearn")
    from georisk.validation.ensemble import SkForest, dump_model, load_model

    rng = np.random.default_rng(3)
    X = rng.normal(size=(200, 4))
    y = ((X[:, 0] > 0.2) ^ (X[:, 1] > 0.5)).astype(int)  # interaction a linear model cannot fit
    sk = SkForest(60).fit(X, y)
    back = load_model(__import__("json").loads(__import__("json").dumps(dump_model("sk_forest", sk))))
    assert np.allclose(back.predict_proba(X), sk.predict_proba(X))  # served copy needs no scikit-learn
    from georisk.validation.metrics import auc

    assert auc(y, back.predict_proba(X)) > 0.9


def test_outside_domain_lists_tasks_the_model_does_not_cover():
    from georisk.calibration import outside_domain

    art = {"fire": {"domain": [32.0, 49.0, -124.0, -104.0]}, "flood": {"domain": [24.0, 43.0, -123.0, -78.0]}}
    assert outside_domain(39.96, -83.0, art) == ["fire"]  # Columbus: flood covered, fire not
    assert outside_domain(38.5, -122.7, art) == []


def test_drivers_point_the_right_way_and_vanish_at_the_typical_value():
    art = fake_artifact()
    wet = extract_features(make_scene("floodplain"))
    out = calibrated_probabilities(wet, T, art)["flood"]
    drivers = {d["label"]: d for d in out["drivers"]}
    assert drivers["Open water"]["effect_pp"] > 0  # lots of open water raises flood probability
    assert out["drivers"][0]["label"] == "Open water"  # and it is the biggest driver
    from georisk.calibration import explain
    from georisk.validation.ensemble import load_model

    spec = art["flood"]
    x = np.array([[spec["fill"]["water_fraction"], spec["fill"]["t_relief_m"]]])
    assert explain(load_model(spec["model"]), spec, x) == []  # a perfectly typical site has nothing to explain


def test_driver_values_are_reported_in_display_units():
    art = fake_artifact()
    f = extract_features(make_scene("floodplain"))
    d = next(x for x in calibrated_probabilities(f, T, art)["flood"]["drivers"] if x["label"] == "Open water")
    p = d["parts"][0]
    assert p["unit"] == "%" and 0 <= p["value"] <= 100 and p["typical"] == 20.0  # fill 0.2 shown as 20 %


def test_correlated_features_are_explained_together_not_as_offsetting_pairs():
    import json

    from georisk.calibration import explain, load_artifact
    from georisk.validation.ensemble import load_model

    spec = load_artifact().get("flood")
    if not spec:
        return
    model, cols = load_model(spec["model"]), spec["features"]
    x = np.array([[spec["fill"][c] for c in cols]])
    x[0, cols.index("t_mean_slope_deg")] *= 4
    x[0, cols.index("t_p90_slope_deg")] *= 4
    labels = [d["label"] for d in explain(model, spec, x, top=8)]
    assert "Terrain steepness" in labels and "Mean slope" not in labels  # one driver, not two that cancel
    assert json.dumps(explain(model, spec, x))  # JSON-serialisable for the API


def test_recalibrated_wrapper_changes_probability_and_drivers_consistently():
    from georisk.calibration import Recalibrated, explain
    from georisk.validation.metrics import Logistic

    rng = np.random.default_rng(4)
    X = rng.random((300, 2))
    y = (X[:, 0] + 0.2 * rng.normal(size=300) > 0.5).astype(int)
    base = Logistic(1.0).fit(X, y)
    cal = Recalibrated(base, {"a": 2.0, "b": 0.0})
    x = np.array([[0.9, 0.5]])
    assert cal.predict_proba(x)[0] > base.predict_proba(x)[0]  # a > 1 stretches an above-0.5 prediction upward
    assert Recalibrated(base, None).predict_proba(x)[0] == base.predict_proba(x)[0]
    spec = {"features": ["water_fraction", "t_relief_m"], "fill": {"water_fraction": 0.5, "t_relief_m": 0.5}}
    d = explain(cal, spec, x)
    assert d and abs(d[0]["effect_pp"]) > 0


def test_similar_sites_lookup_reports_the_matching_band():
    from georisk.calibration import similar_sites

    spec = {"reliability": [{"lo": 0.0, "hi": 0.5, "n": 200, "rate": 0.2, "ci": [0.15, 0.26]},
                            {"lo": 0.5, "hi": 1.0, "n": 150, "rate": 0.7, "ci": [0.62, 0.77]}]}
    assert similar_sites(spec, 0.61) == {"n": 150, "observed_pct": 70, "ci95_pct": [62, 77], "band_pct": [50, 100]}
    assert similar_sites(spec, 1.0)["observed_pct"] == 70 and similar_sites({}, 0.4) is None


def test_memo_cites_the_track_record_with_numbers_that_pass_the_guardrail():
    from georisk import analyst

    cal = {"flood": {"probability": 0.61, "meaning": "m", "drivers": [],
                     "evidence": {"n": 150, "observed_pct": 70, "ci95_pct": [62, 77], "band_pct": [50, 100]}}}
    f = analyst.build_facts(cal, overall=30.0, overall_level="low", confidence="high", acquired=None, cloud_cover=None, warnings=[], source="x")
    text = analyst.template_memo(f)
    assert "70% of 150 held-out sites" in text and "95% range 62-77%" in text
    assert analyst.verify_numbers(text, f)  # the memo's own numbers all come from the facts


def test_opportunities_do_not_trigger_the_unvalidated_heuristic_warning():
    from georisk.calibration import outside_domain

    art = {"fire": {"kind": "hazard", "domain": [32.0, 49.0, -124.0, -104.0]},
           "solar": {"kind": "opportunity", "domain": [32.0, 49.0, -124.0, -104.0]}}
    assert outside_domain(39.96, -83.0, art) == ["fire"]  # solar has no heuristic score to call unvalidated


def test_calibrated_output_carries_the_kind_so_the_ui_and_memo_can_treat_it_differently():
    spec = fake_artifact()["flood"] | {"kind": "opportunity"}
    out = calibrated_probabilities(extract_features(make_scene("floodplain")), T, {"flood": spec})
    assert out["flood"]["kind"] == "opportunity"
    assert calibrated_probabilities(extract_features(make_scene("floodplain")), T, fake_artifact())["flood"]["kind"] == "hazard"  # default


def test_solar_is_withheld_without_grid_data_instead_of_guessing_a_median():
    from georisk.grid import GridFeatures

    spec = fake_artifact()["flood"] | {"kind": "opportunity", "features": ["water_fraction", "g_dist_km"],
                                         "fill": {"water_fraction": 0.2, "g_dist_km": 12.0}}
    # model trained on a column order that includes the grid feature
    rng = np.random.default_rng(9)
    X = rng.random((200, 2))
    y = (X[:, 1] < 0.5).astype(int)
    from georisk.validation.metrics import Logistic

    spec["model"] = Logistic(1.0).fit(X, y).to_dict()
    f = extract_features(make_scene("floodplain"))
    assert calibrated_probabilities(f, T, {"solar": spec}) == {}  # no grid -> no probability
    out = calibrated_probabilities(f, T, {"solar": spec}, grid=GridFeatures(0.3, 0.3, 4.0, 345.0))
    assert out["solar"]["kind"] == "opportunity" and 0 <= out["solar"]["probability"] <= 1


def test_the_experimental_caveat_travels_with_the_probability():
    spec = fake_artifact()["flood"] | {"kind": "opportunity", "caveat": "Experimental: rough screen."}
    out = calibrated_probabilities(extract_features(make_scene("floodplain")), T, {"flood": spec})
    assert out["flood"]["caveat"] == "Experimental: rough screen."
    assert calibrated_probabilities(extract_features(make_scene("floodplain")), T, fake_artifact())["flood"]["caveat"] is None


WEAK = {"weak_regions": [{"name": "new_orleans", "auc": 0.54, "n": 93, "shape": "circle", "geometry": [29.95, -90.07, 25]},
                         {"name": "az_central", "auc": 0.58, "n": 41, "shape": "box", "geometry": [33.5, 35.0, -112.5, -110.5]}]}


def test_region_caution_fires_inside_a_weak_circle_or_box_and_nowhere_else():
    from georisk.calibration import region_caution

    assert "New Orleans" in region_caution(WEAK, 29.95, -90.07) and "0.54" in region_caution(WEAK, 30.0, -90.0)
    assert "central Arizona" in region_caution(WEAK, 34.0, -111.5)
    assert region_caution(WEAK, 30.5, -90.07) is None  # ~60 km north: outside the 25 km circle
    assert region_caution(WEAK, 36.0, -111.5) is None  # outside the box
    assert region_caution(WEAK, None, None) is None and region_caution({}, 29.95, -90.07) is None


def test_the_caution_reaches_the_output_and_the_memo_with_numbers_that_pass_the_guardrail():
    from georisk import analyst

    spec = fake_artifact()["flood"] | WEAK
    f = extract_features(make_scene("floodplain"))
    out = calibrated_probabilities(f, T, {"flood": spec}, lat=29.95, lon=-90.07)
    assert "extra caution" in out["flood"]["region_caution"]
    assert calibrated_probabilities(f, T, {"flood": spec}, lat=39.0, lon=-83.0)["flood"]["region_caution"] is None
    facts = analyst.build_facts(out, overall=30.0, overall_level="low", confidence="high", acquired=None, cloud_cover=None, warnings=[], source="x")
    text = analyst.template_memo(facts)
    assert "Caution for flood: This model tested weakly near New Orleans (held-out AUC 0.54 on 93 sites)" in text
    assert analyst.verify_numbers(text, facts)
