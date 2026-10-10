import mne
import numpy as np
import pytest

from schizo_eeg.config import CHANNELS


def make_raw(seconds=30.0, sfreq=250.0, channels=None, seed=0, alpha_amp=0.0):
    """Synthetic recording: white noise plus an optional 10 Hz alpha rhythm."""
    channels = list(channels or CHANNELS)
    rng = np.random.default_rng(seed)
    n = int(seconds * sfreq)
    t = np.arange(n) / sfreq
    data = rng.normal(scale=10e-6, size=(len(channels), n))
    # Vary amplitude across channels; a signal identical on every channel would
    # be removed entirely by the average reference.
    gains = np.linspace(0.0, 2.0, len(channels))[:, None]
    data += alpha_amp * gains * np.sin(2 * np.pi * 10 * t)
    info = mne.create_info(channels, sfreq, ch_types="eeg")
    return mne.io.RawArray(data, info, verbose="ERROR")


@pytest.fixture
def raw():
    return make_raw()
