"""Model registry.

Every model is built as a scikit-learn compatible estimator that takes raw
windows ``(n_windows, n_channels, n_times)`` and exposes ``predict_proba``, so
evaluation, training and inference treat classical and deep models the same.
"""

from __future__ import annotations

from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

from ..config import ExperimentConfig
from .classical import CLASSIFIERS, build_classical_pipeline

DEEP_MODELS = ("cnn1d", "chrononet", "cnn_gru")
AVAILABLE_MODELS = tuple(CLASSIFIERS) + DEEP_MODELS


def build_estimator(config: ExperimentConfig) -> BaseEstimator:
    model = config.model
    pre = config.preprocessing

    if model.name in CLASSIFIERS:
        estimator = build_classical_pipeline(
            model.name, model.params, model.features, pre.sfreq, config.seed
        )
    elif model.name in DEEP_MODELS:
        from .deep import KerasWindowClassifier  # TensorFlow is optional

        estimator = KerasWindowClassifier(
            architecture=model.name, seed=config.seed, **model.params
        )
    else:
        raise ValueError(f"Unknown model {model.name!r}; choose from {AVAILABLE_MODELS}")

    if model.param_grid:
        # Hyperparameters are tuned on the training subjects only (nested CV
        # when called inside an outer evaluation loop). Pass groups= to fit.
        estimator = GridSearchCV(
            estimator,
            model.param_grid,
            # Stratified so every inner training fold contains both classes.
            cv=StratifiedGroupKFold(
                n_splits=config.evaluation.n_splits, shuffle=True, random_state=config.seed
            ),
            scoring="roc_auc",
            refit=True,
        )
    return estimator
