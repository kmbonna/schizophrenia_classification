"""Hand-crafted features computed per window and channel.

Each extractor is a stateless scikit-learn transformer mapping windows of shape
``(n_windows, n_channels, n_times)`` to a 2-D feature matrix, so it can sit at
the front of a ``Pipeline`` and be saved with the model.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import signal, stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion

# Statistics from the original notebook, in the same order.
TIME_DOMAIN_FUNCS = {
    "mean": lambda x: np.mean(x, axis=-1),
    "std": lambda x: np.std(x, axis=-1),
    "ptp": lambda x: np.ptp(x, axis=-1),
    "var": lambda x: np.var(x, axis=-1),
    "min": lambda x: np.min(x, axis=-1),
    "max": lambda x: np.max(x, axis=-1),
    "argmin": lambda x: np.argmin(x, axis=-1),
    "argmax": lambda x: np.argmax(x, axis=-1),
    "mean_square": lambda x: np.mean(x ** 2, axis=-1),
    "rms": lambda x: np.sqrt(np.mean(x ** 2, axis=-1)),
    "abs_diff_sum": lambda x: np.sum(np.abs(np.diff(x, axis=-1)), axis=-1),
    "skew": lambda x: stats.skew(x, axis=-1),
    "kurtosis": lambda x: stats.kurtosis(x, axis=-1),
}

FREQUENCY_BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def _check_windows(X) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError(f"Expected (n_windows, n_channels, n_times), got {X.shape}")
    return X


class TimeDomainFeatures(BaseEstimator, TransformerMixin):
    """Summary statistics of each channel's signal.

    Output columns are grouped by feature then channel:
    ``[mean_ch0 .. mean_chN, std_ch0 .. std_chN, ...]``.
    """

    def __init__(self, features: Optional[Sequence[str]] = None):
        self.features = features

    def fit(self, X, y=None):
        unknown = set(self._names) - set(TIME_DOMAIN_FUNCS)
        if unknown:
            raise ValueError(f"Unknown time-domain features: {sorted(unknown)}")
        return self

    @property
    def _names(self) -> List[str]:
        return list(self.features or TIME_DOMAIN_FUNCS)

    def transform(self, X) -> np.ndarray:
        X = _check_windows(X)
        return np.concatenate(
            [TIME_DOMAIN_FUNCS[name](X) for name in self._names], axis=-1
        ).astype(np.float64)


class BandPowerFeatures(BaseEstimator, TransformerMixin):
    """Power in canonical EEG frequency bands, estimated with Welch's method.

    With ``relative=True`` each band is divided by total power across the
    bands, which removes amplitude differences between recordings. Values are
    log-transformed to make them closer to normally distributed.
    """

    def __init__(
        self,
        sfreq: float = 250.0,
        bands: Optional[Dict[str, Tuple[float, float]]] = None,
        relative: bool = True,
        log: bool = True,
        window_seconds: float = 2.0,
    ):
        self.sfreq = sfreq
        self.bands = bands
        self.relative = relative
        self.log = log
        self.window_seconds = window_seconds

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> np.ndarray:
        X = _check_windows(X)
        bands = self.bands or FREQUENCY_BANDS
        nperseg = min(X.shape[-1], int(self.window_seconds * self.sfreq))
        freqs, psd = signal.welch(X, fs=self.sfreq, nperseg=nperseg, axis=-1)

        df = freqs[1] - freqs[0]
        powers = np.stack(
            [
                psd[..., (freqs >= lo) & (freqs < hi)].sum(axis=-1) * df
                for lo, hi in bands.values()
            ],
            axis=1,
        )  # (n_windows, n_bands, n_channels)
        if self.relative:
            powers = powers / powers.sum(axis=1, keepdims=True)
        if self.log:
            powers = np.log(powers + np.finfo(float).tiny)
        return powers.reshape(len(X), -1)


def build_feature_extractor(
    kinds: Sequence[str] = ("time",), sfreq: float = 250.0
) -> FeatureUnion:
    """Combine feature families by name: ``"time"`` and/or ``"bandpower"``."""
    available = {
        "time": lambda: TimeDomainFeatures(),
        "bandpower": lambda: BandPowerFeatures(sfreq=sfreq),
    }
    unknown = set(kinds) - set(available)
    if unknown:
        raise ValueError(f"Unknown feature kinds {sorted(unknown)}; use {list(available)}")
    return FeatureUnion([(kind, available[kind]()) for kind in kinds])
