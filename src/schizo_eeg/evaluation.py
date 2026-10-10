"""Subject-grouped cross-validation with subject-level metrics.

Windows from one subject are never split across train and test. Each held-out
subject is classified by averaging the probabilities of all their windows,
which is the decision a clinician would actually care about.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold

from .config import EvaluationConfig
from .data import WindowDataset
from .utils import fit_with_groups

log = logging.getLogger(__name__)


def aggregate_by_subject(
    window_proba: np.ndarray, groups: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Mean window probability per subject -> ``(subject_indices, probabilities)``."""
    subjects = np.unique(groups)
    return subjects, np.array([window_proba[groups == s].mean() for s in subjects])


def wilson_interval(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson score interval for a proportion (sensible for small n)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (float(centre - half), float(centre + half))


def binary_metrics(y_true, proba, threshold: float) -> Dict[str, Any]:
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(proba) > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    correct = int(tp + tn)
    metrics = {
        "n": int(len(y_true)),
        "accuracy": accuracy_score(y_true, y_pred),
        "accuracy_95ci": wilson_interval(correct, len(y_true)),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
        "specificity": tn / (tn + fp) if tn + fp else float("nan"),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
    metrics["roc_auc"] = (
        roc_auc_score(y_true, proba) if len(np.unique(y_true)) == 2 else float("nan")
    )
    return metrics


@dataclass
class CVResult:
    subject_ids: List[str]
    subject_labels: np.ndarray
    subject_proba: np.ndarray
    window_labels: np.ndarray
    window_proba: np.ndarray
    threshold: float
    folds: List[List[str]] = field(default_factory=list)

    @property
    def subject_metrics(self) -> Dict[str, Any]:
        return binary_metrics(self.subject_labels, self.subject_proba, self.threshold)

    @property
    def window_metrics(self) -> Dict[str, Any]:
        return binary_metrics(self.window_labels, self.window_proba, 0.5)

    def summary(self) -> Dict[str, Any]:
        return {
            "subject_level": self.subject_metrics,
            "window_level": self.window_metrics,
            "per_subject": [
                {
                    "subject": sid,
                    "label": int(label),
                    "probability": float(p),
                    "correct": bool((p > self.threshold) == label),
                }
                for sid, label, p in zip(
                    self.subject_ids, self.subject_labels, self.subject_proba
                )
            ],
            "folds": self.folds,
        }


def make_splitter(config: EvaluationConfig, seed: int):
    if config.cv == "logo":
        return LeaveOneGroupOut()
    if config.cv == "group_kfold":
        return StratifiedGroupKFold(n_splits=config.n_splits, shuffle=True, random_state=seed)
    raise ValueError(f"Unknown cv {config.cv!r}; use 'logo' or 'group_kfold'")


def cross_validate(
    make_estimator: Callable[[], BaseEstimator],
    dataset: WindowDataset,
    config: EvaluationConfig,
    seed: int = 42,
) -> CVResult:
    """Fit a fresh estimator per fold and collect out-of-fold predictions."""
    splitter = make_splitter(config, seed)
    window_proba = np.full(len(dataset.y), np.nan)
    folds = []

    splits = list(splitter.split(dataset.X, dataset.y, groups=dataset.groups))
    for i, (train, test) in enumerate(splits, start=1):
        test_subjects = [dataset.subject_ids[g] for g in np.unique(dataset.groups[test])]
        log.info("Fold %d/%d: testing on %s", i, len(splits), ", ".join(test_subjects))

        estimator = make_estimator()
        fit_with_groups(estimator, dataset.X[train], dataset.y[train], dataset.groups[train])
        window_proba[test] = estimator.predict_proba(dataset.X[test])[:, 1]
        folds.append(test_subjects)

    subjects, subject_proba = aggregate_by_subject(window_proba, dataset.groups)
    return CVResult(
        subject_ids=[dataset.subject_ids[s] for s in subjects],
        subject_labels=dataset.subject_labels[subjects],
        subject_proba=subject_proba,
        window_labels=dataset.y,
        window_proba=window_proba,
        threshold=config.threshold,
        folds=folds,
    )
