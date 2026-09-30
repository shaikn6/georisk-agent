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
