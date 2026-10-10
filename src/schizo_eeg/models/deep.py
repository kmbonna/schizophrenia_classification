"""Deep models over raw windows, wrapped as scikit-learn estimators.

The wrapper owns its per-channel scaler, so a fitted ``KerasWindowClassifier``
is a complete windows -> probability step and pickles (with joblib) like any
other estimator.
"""

from __future__ import annotations

import os
import tempfile
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402
from tensorflow.keras import layers  # noqa: E402


class ChannelScaler(BaseEstimator, TransformerMixin):
    """Standardise each channel using mean/std over all training windows and samples."""

    def fit(self, X, y=None):
        X = np.asarray(X)
        self.mean_ = X.mean(axis=(0, 2), keepdims=True)
        self.std_ = X.std(axis=(0, 2), keepdims=True) + 1e-12
        return self

    def transform(self, X):
        return ((np.asarray(X) - self.mean_) / self.std_).astype(np.float32)


# --- architectures -----------------------------------------------------------
# Inputs are (n_times, n_channels): Keras Conv1D expects channels last.


def cnn1d(input_shape: Tuple[int, int]) -> tf.keras.Model:
    """Compact 1D CNN (the notebook's first ``cnnmodel``)."""
    return tf.keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv1D(10, kernel_size=8, strides=2),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.AveragePooling1D(pool_size=2, strides=2),
            layers.Dropout(0.35),
            layers.Conv1D(5, kernel_size=5, strides=1),
            layers.LeakyReLU(),
            layers.AveragePooling1D(pool_size=3, strides=1),
            layers.Flatten(),
            layers.Dense(128),
            layers.LeakyReLU(),
            layers.Dropout(0.4),
            layers.Dense(32),
            layers.LeakyReLU(),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="cnn1d",
    )


def _inception_block(x, filters: int = 32):
    """Parallel convolutions with exponentially growing kernels (2, 4, 8)."""
    branches = [
        layers.Conv1D(filters, 2, strides=2, activation="relu", padding="same")(x),
        layers.Conv1D(filters, 4, strides=2, activation="relu", padding="causal")(x),
        layers.Conv1D(filters, 8, strides=2, activation="relu", padding="causal")(x),
    ]
    return layers.Concatenate(axis=-1)(branches)


def chrononet(input_shape: Tuple[int, int]) -> tf.keras.Model:
    """ChronoNet (Roy et al., 2018): inception-style 1D convs + densely connected GRUs."""
    inputs = layers.Input(shape=input_shape)
    x = _inception_block(inputs)
    x = _inception_block(x)
    x = _inception_block(x)
    x = layers.BatchNormalization()(x)

    gru1 = layers.GRU(32, return_sequences=True)(x)
    gru2 = layers.GRU(32, return_sequences=True)(gru1)
    gru3 = layers.GRU(32, return_sequences=True)(layers.Concatenate()([gru1, gru2]))
    gru4 = layers.GRU(32)(layers.Concatenate()([gru1, gru2, gru3]))
    outputs = layers.Dense(1, activation="sigmoid")(gru4)
    return tf.keras.Model(inputs, outputs, name="chrononet")


def cnn_gru(input_shape: Tuple[int, int]) -> tf.keras.Model:
    """Simpler sequential CNN-GRU (the notebook's ``gru_cnn_model``)."""
    return tf.keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv1D(32, 2, strides=2, activation="relu", padding="same"),
            layers.Conv1D(32, 4, strides=2, activation="relu", padding="causal"),
            layers.Conv1D(32, 8, strides=2, activation="relu", padding="causal"),
            layers.BatchNormalization(),
            layers.GRU(32, return_sequences=True),
            layers.GRU(32, return_sequences=True),
            layers.GRU(32),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="cnn_gru",
    )


ARCHITECTURES: Dict[str, Tuple[Callable[[Tuple[int, int]], tf.keras.Model], float]] = {
    # name: (builder, default learning rate)
    "cnn1d": (cnn1d, 5e-4),
    "chrononet": (chrononet, 1e-3),
    "cnn_gru": (cnn_gru, 5e-4),
}


# --- estimator ---------------------------------------------------------------


class KerasWindowClassifier(BaseEstimator, ClassifierMixin):
    """Binary classifier over windows ``(n_windows, n_channels, n_times)``."""

    def __init__(
        self,
        architecture: str = "cnn1d",
        epochs: int = 10,
        batch_size: int = 128,
        learning_rate: Optional[float] = None,
        seed: int = 42,
        verbose: int = 0,
    ):
        self.architecture = architecture
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed
        self.verbose = verbose

    @staticmethod
    def _to_keras(X: np.ndarray) -> np.ndarray:
        return np.moveaxis(X, 1, 2)  # (n, channels, times) -> (n, times, channels)

    def fit(self, X, y):
        if self.architecture not in ARCHITECTURES:
            raise ValueError(f"Unknown architecture {self.architecture!r}")
        builder, default_lr = ARCHITECTURES[self.architecture]

        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(self.seed)

        self.classes_ = np.array([0, 1])
        self.scaler_ = ChannelScaler().fit(X)
        X_in = self._to_keras(self.scaler_.transform(X))

        self.model_ = builder(X_in.shape[1:])
        self.model_.compile(
            optimizer=tf.keras.optimizers.Adam(self.learning_rate or default_lr),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        self.history_ = self.model_.fit(
            X_in,
            np.asarray(y, dtype=np.float32),
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=True,
            verbose=self.verbose,
        ).history
        return self

    def predict_proba(self, X) -> np.ndarray:
        X_in = self._to_keras(self.scaler_.transform(X))
        p = self.model_.predict(X_in, batch_size=self.batch_size, verbose=0).ravel()
        return np.column_stack([1 - p, p])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

    # Keras models do not pickle reliably across TF versions; store HDF5 bytes.
    def __getstate__(self):
        state = self.__dict__.copy()
        model = state.pop("model_", None)
        if model is not None:
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "model.h5")
                model.save(path)
                with open(path, "rb") as f:
                    state["_model_h5"] = f.read()
        return state

    def __setstate__(self, state):
        model_bytes = state.pop("_model_h5", None)
        self.__dict__.update(state)
        if model_bytes is not None:
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "model.h5")
                with open(path, "wb") as f:
                    f.write(model_bytes)
                self.model_ = tf.keras.models.load_model(path, compile=False)
