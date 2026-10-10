"""Turn a continuous EEG recording into fixed-length, filtered windows.

This is the first stage of the pipeline and is applied identically during
training and inference.
"""

from __future__ import annotations

import mne
import numpy as np

from .config import PreprocessingConfig


class EEGPreprocessor:
    """Re-reference, band-pass filter and window an ``mne.io.Raw`` recording.

    Output shape is ``(n_windows, n_channels, n_times)`` with channels in the
    order given by ``config.channels``.
    """

    def __init__(self, config: PreprocessingConfig):
        self.config = config

    def transform(self, raw: mne.io.BaseRaw) -> np.ndarray:
        cfg = self.config
        raw = raw.copy().load_data(verbose="ERROR")

        missing = [ch for ch in cfg.channels if ch not in raw.ch_names]
        if missing:
            raise ValueError(f"Recording is missing required channels: {missing}")
        raw.pick(cfg.channels)
        raw.reorder_channels(cfg.channels)

        if not np.isclose(raw.info["sfreq"], cfg.sfreq):
            raw.resample(cfg.sfreq, verbose="ERROR")

        raw.set_eeg_reference(cfg.reference, verbose="ERROR")
        raw.filter(l_freq=cfg.l_freq, h_freq=cfg.h_freq, verbose="ERROR")

        epochs = mne.make_fixed_length_epochs(
            raw,
            duration=cfg.epoch_duration,
            overlap=cfg.epoch_overlap,
            preload=True,
            verbose="ERROR",
        )
        windows = epochs.get_data(copy=False).astype(np.float32)
        if windows.shape[0] == 0:
            raise ValueError(
                f"Recording is shorter than one {cfg.epoch_duration}s window."
            )
        return windows
