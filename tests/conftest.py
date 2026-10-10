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
    data += alpha_amp * np.sin(2 * np.pi * 10 * t)
    info = mne.create_info(channels, sfreq, ch_types="eeg")
    return mne.io.RawArray(data, info, verbose="ERROR")


@pytest.fixture
def raw():
    return make_raw()
