import numpy as np

from georisk.foundation import chips
from georisk.foundation.evaluate import EmbModel
from georisk.validation.metrics import auc


def test_chip_key_is_stable_and_filesystem_safe():
    assert chips.key(39.956, -83.018) == "39.95600_-83.01800"


def test_collect_isolates_failures_and_skips_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(chips, "CACHE", tmp_path)
    calls = []

    def fake(lat, lon, **k):
        calls.append((lat, lon))
        if lat == 2.0:
            raise RuntimeError("no clear scene")
        return np.ones((6, 4, 4), dtype="float32")

    monkeypatch.setattr(chips, "fetch_chip", fake)
    first = chips.collect([(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)], workers=1)
    assert first[chips.key(1.0, 1.0)] == "ok" and first[chips.key(3.0, 3.0)] == "ok"
    assert first[chips.key(2.0, 2.0)].startswith("error")  # one bad point does not stop the run
    calls.clear()
    chips.collect([(1.0, 1.0), (3.0, 3.0)], workers=1)
    assert calls == []  # cached chips are not refetched


def test_embedding_model_learns_signal_and_never_sees_test_rows_when_fitting():
    rng = np.random.default_rng(0)
    n, d = 300, 40
    latent = rng.normal(size=(n, 3))  # like real embeddings: a few shared factors spread over many correlated dims
    Z = latent @ rng.normal(size=(3, d)) + 0.3 * rng.normal(size=(n, d))
    y = (latent[:, 0] * 2 + latent[:, 1] + rng.normal(scale=0.7, size=n) > 0).astype(int)
    F = rng.normal(size=(n, 3))
    X = np.hstack([F, Z])
    tr, te = slice(0, 200), slice(200, 300)
    m = EmbModel(3, k=8, base="logistic").fit(X[tr], y[tr])
    assert auc(y[te], m.predict_proba(X[te])) > 0.85
    m2 = EmbModel(3, k=8, base="logistic", use_feat=True).fit(X[tr], y[tr])
    assert m2._design(X[te]).shape[1] == 3 + 8  # features + PCA components
    mu_before = m.mu.copy()
    m.predict_proba(X[te])
    assert np.array_equal(mu_before, m.mu)  # predicting never updates the fitted PCA statistics
