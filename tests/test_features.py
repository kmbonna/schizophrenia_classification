import numpy as np
import pytest

from schizo_eeg.features import (
    FREQUENCY_BANDS,
    TIME_DOMAIN_FUNCS,
    BandPowerFeatures,
    TimeDomainFeatures,
    build_feature_extractor,
)

SFREQ = 250.0


def sine_windows(freq, n_windows=3, n_channels=4, seconds=5.0):
    t = np.arange(int(seconds * SFREQ)) / SFREQ
    wave = np.sin(2 * np.pi * freq * t)
    return np.broadcast_to(wave, (n_windows, n_channels, t.size)).copy()


def test_time_domain_shape_and_values():
    X = np.random.default_rng(0).normal(size=(5, 3, 100))
    out = TimeDomainFeatures().fit_transform(X)
    assert out.shape == (5, 3 * len(TIME_DOMAIN_FUNCS))
    # First block is the per-channel mean, second the std.
    np.testing.assert_allclose(out[:, :3], X.mean(axis=-1))
    np.testing.assert_allclose(out[:, 3:6], X.std(axis=-1))


def test_time_domain_feature_subset():
    X = np.ones((2, 3, 10))
    out = TimeDomainFeatures(features=["mean", "max"]).fit_transform(X)
    assert out.shape == (2, 6)


def test_time_domain_unknown_feature():
    with pytest.raises(ValueError, match="Unknown"):
        TimeDomainFeatures(features=["nope"]).fit(np.ones((1, 1, 4)))


@pytest.mark.parametrize("freq,band", [(2, "delta"), (6, "theta"), (10, "alpha"), (20, "beta"), (40, "gamma")])
def test_band_power_peaks_in_correct_band(freq, band):
    feats = BandPowerFeatures(sfreq=SFREQ, log=False).fit_transform(sine_windows(freq))
    n_channels = 4
    per_band = feats.reshape(3, len(FREQUENCY_BANDS), n_channels)
    assert list(FREQUENCY_BANDS)[per_band[0, :, 0].argmax()] == band
    # Relative power sums to one across bands.
    np.testing.assert_allclose(per_band.sum(axis=1), 1.0, rtol=1e-6)


def test_build_feature_extractor_combines_families():
    X = sine_windows(10)
    out = build_feature_extractor(["time", "bandpower"], sfreq=SFREQ).fit_transform(X)
    assert out.shape == (3, 4 * (len(TIME_DOMAIN_FUNCS) + len(FREQUENCY_BANDS)))
    with pytest.raises(ValueError):
        build_feature_extractor(["wavelets"])
