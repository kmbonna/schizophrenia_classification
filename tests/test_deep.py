import pickle

import numpy as np
import pytest

pytest.importorskip("tensorflow")

from schizo_eeg.models.deep import ARCHITECTURES, ChannelScaler, KerasWindowClassifier  # noqa: E402


def windows(n=32, channels=4, times=128, seed=0):
    rng = np.random.default_rng(seed)
    y = np.arange(n) % 2
    X = rng.normal(size=(n, channels, times)) * (1 + 2 * y[:, None, None])
    return X.astype(np.float32), y


def test_channel_scaler_standardises_per_channel():
    X, _ = windows()
    Z = ChannelScaler().fit_transform(X * 5 + 3)
    np.testing.assert_allclose(Z.mean(axis=(0, 2)), 0, atol=1e-5)
    np.testing.assert_allclose(Z.std(axis=(0, 2)), 1, atol=1e-4)


@pytest.mark.parametrize("architecture", sorted(ARCHITECTURES))
def test_fit_predict_and_pickle_roundtrip(architecture):
    X, y = windows()
    clf = KerasWindowClassifier(architecture=architecture, epochs=1, batch_size=16).fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (len(X), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1, rtol=1e-6)

    restored = pickle.loads(pickle.dumps(clf))
    np.testing.assert_allclose(restored.predict_proba(X), proba, rtol=1e-5)
