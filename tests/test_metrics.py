import numpy as np
import pytest

from georisk.validation.metrics import Logistic, auc, brier, cross_validate, stratified_kfold


def test_auc_known_cases():
    assert auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)  # all ties
    assert auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.75)  # textbook example
    assert np.isnan(auc([1, 1, 1], [0.1, 0.2, 0.3]))


def test_brier():
    assert brier([1, 0], [1.0, 0.0]) == 0.0
    assert brier([1, 0], [0.5, 0.5]) == pytest.approx(0.25)


def make_data(n=400, seed=1, signal=2.0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    p = 1 / (1 + np.exp(-(signal * X[:, 0] - 0.5)))
    return X, (rng.random(n) < p).astype(int)


def test_logistic_recovers_signal_and_calibrates():
    X, y = make_data()
    m = Logistic(l2=0.1).fit(X, y)
    p = m.predict_proba(X)
    assert auc(y, p) > 0.85
    assert abs(p.mean() - y.mean()) < 0.03  # calibrated in the large
    assert abs(m.w[0]) > 5 * abs(m.w[1])  # weight concentrates on the informative feature


def test_logistic_roundtrip_serialisation():
    X, y = make_data()
    m = Logistic().fit(X, y)
    m2 = Logistic.from_dict(m.to_dict())
    assert np.allclose(m.predict_proba(X), m2.predict_proba(X))


def test_cross_validation_is_honest_on_pure_noise():
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(120, 4)), rng.integers(0, 2, 120)
    cv = cross_validate(X, y)
    assert 0.3 < cv["auc"] < 0.7  # cannot beat chance without signal
    assert cv["auc_ci95"][0] < 0.55 < cv["auc_ci95"][1] + 0.2


def test_cross_validation_finds_real_signal():
    X, y = make_data()
    assert cross_validate(X, y)["auc"] > 0.8


def test_stratified_folds_partition_and_keep_both_classes():
    y = np.array([0] * 30 + [1] * 20)
    seen = []
    for tr, te in stratified_kfold(y, 5):
        assert set(tr).isdisjoint(te) and set(y[te]) == {0, 1}
        seen += te.tolist()
    assert sorted(seen) == list(range(50))


def test_leave_group_out_generalises_across_groups_and_exposes_leakage():
    from georisk.validation.metrics import leave_group_out

    rng = np.random.default_rng(3)
    groups = np.repeat(np.arange(6), 40)
    # feature 0 carries real signal everywhere
    X = rng.normal(size=(240, 3))
    y = (rng.random(240) < 1 / (1 + np.exp(-2 * X[:, 0]))).astype(int)
    r = leave_group_out(X, y, groups)
    assert r["auc"] > 0.8 and r["n"] == 240 and len(r["per_group_auc"]) == 6

    # Spatial leakage scenario: each region has its own base rate and the features can identify the region
    # (one-hot). Random K-fold memorises region -> rate and looks great; leave-region-out cannot.
    onehot = np.eye(6)[groups]
    rate = np.where(groups < 3, 0.9, 0.1)
    y_leak = (rng.random(240) < rate).astype(int)
    fooled = cross_validate(onehot, y_leak, k=5)["auc"]
    honest = leave_group_out(onehot, y_leak, groups)["auc"]
    assert fooled > 0.85 and honest < 0.65


def test_ece_is_zero_for_perfectly_calibrated_and_large_for_overconfident():
    from georisk.validation.metrics import ece

    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 20000)
    y = (rng.random(20000) < p).astype(int)
    assert ece(y, p) < 0.02
    assert ece(y, np.clip(p * 0.3 + 0.7, 0, 1)) > 0.25  # everything pushed to ~0.7-1.0 while outcomes are unchanged


def test_platt_scaling_recovers_a_squeezed_model_and_keeps_the_ranking():
    from georisk.validation.metrics import auc, ece, platt_apply, platt_fit

    rng = np.random.default_rng(1)
    true_p = rng.uniform(0.02, 0.98, 6000)
    y = (rng.random(6000) < true_p).astype(int)
    squeezed = 0.5 + (true_p - 0.5) * 0.4  # a model that compresses everything toward 0.5
    a, b = platt_fit(squeezed, y)
    fixed = platt_apply(squeezed, a, b)
    assert a > 1.5  # the fit stretches the squeezed probabilities back out
    assert ece(y, fixed) < ece(y, squeezed) / 3
    assert np.isclose(auc(y, fixed), auc(y, squeezed), atol=1e-9)  # monotone: ranking is untouched


def test_wilson_interval_matches_known_values_and_handles_edges():
    from georisk.validation.metrics import wilson

    lo, hi = wilson(50, 100)
    assert abs(lo - 0.4038) < 1e-3 and abs(hi - 0.5962) < 1e-3
    assert wilson(0, 20)[0] == 0.0 and wilson(20, 20)[1] == 1.0 and wilson(0, 0) == (0.0, 1.0)


def test_reliability_table_reports_observed_rates_and_merges_thin_bands():
    from georisk.validation.metrics import reliability_table

    rng = np.random.default_rng(2)
    p = rng.uniform(0, 1, 3000)
    y = (rng.random(3000) < p).astype(int)
    tab = reliability_table(p, y, min_n=100)
    assert all(b["n"] >= 100 for b in tab) and sum(b["n"] for b in tab) == 3000
    for b in tab:  # for a calibrated model the observed rate tracks the mean prediction inside the band
        inside = (p >= b["lo"]) & (p < (b["hi"] if b["hi"] < 1.0 else 1.0001))
        m = p[inside].mean()
        assert abs(b["rate"] - m) < 4 * (m * (1 - m) / b["n"]) ** 0.5 + 0.01  # within 4 standard errors of sampling noise
    sparse = reliability_table(np.array([0.1, 0.9] * 30), np.array([0, 1] * 30), min_n=40)
    assert sum(b["n"] for b in sparse) == 60 and all(b["n"] >= 40 or len(sparse) == 1 for b in sparse)
