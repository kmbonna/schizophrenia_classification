import mne
import numpy as np
import pytest

from conftest import make_raw
from schizo_eeg.config import CHANNELS, PreprocessingConfig
from schizo_eeg.preprocessing import EEGPreprocessor


def test_window_shape(raw):
    cfg = PreprocessingConfig(epoch_duration=5.0, epoch_overlap=1.0)
    windows = EEGPreprocessor(cfg).transform(raw)
    # 30 s with a 4 s step -> floor((30 - 5) / 4) + 1 = 7 windows
    assert windows.shape == (7, len(CHANNELS), 1250)
    assert windows.dtype == np.float32


def test_channels_are_reordered_by_name():
    raw_a = make_raw()
    # Same signals per channel name, stored in reverse order (like the EDFs).
    order = list(range(len(CHANNELS)))[::-1]
    info = mne.create_info([CHANNELS[i] for i in order], 250.0, ch_types="eeg")
    raw_b = mne.io.RawArray(raw_a.get_data()[order], info, verbose="ERROR")

    pre = EEGPreprocessor(PreprocessingConfig())
    np.testing.assert_allclose(pre.transform(raw_a), pre.transform(raw_b), rtol=1e-5)


def test_missing_channel_raises():
    raw = make_raw(channels=CHANNELS[:-1])
    with pytest.raises(ValueError, match="missing required channels"):
        EEGPreprocessor(PreprocessingConfig()).transform(raw)


def test_resamples_to_configured_rate():
    raw = make_raw(sfreq=500.0)
    windows = EEGPreprocessor(PreprocessingConfig(sfreq=250.0)).transform(raw)
    assert windows.shape[-1] == 1250


def test_input_raw_is_not_modified(raw):
    before = raw.get_data().copy()
    EEGPreprocessor(PreprocessingConfig()).transform(raw)
    np.testing.assert_array_equal(raw.get_data(), before)
