import numpy as np
import pytest

from schizo_eeg.config import EvaluationConfig, ExperimentConfig, ModelConfig
from schizo_eeg.data import WindowDataset
from schizo_eeg.evaluation import (
    aggregate_by_subject,
    binary_metrics,
    cross_validate,
    wilson_interval,
)
from schizo_eeg.models import build_estimator


def toy_dataset(n_subjects=8, windows=6, channels=3, times=250, seed=0):
    """Patients have higher-amplitude signals, so the task is learnable."""
    rng = np.random.default_rng(seed)
    X, y, groups = [], [], []
    for s in range(n_subjects):
        label = s % 2
        X.append(rng.normal(scale=1 + 2 * label, size=(windows, channels, times)))
        y += [label] * windows
        groups += [s] * windows
    return WindowDataset(
        X=np.concatenate(X), y=np.array(y), groups=np.array(groups),
        subject_ids=[f"sub{s}" for s in range(n_subjects)],
    )


def test_aggregate_by_subject():
    subjects, proba = aggregate_by_subject(np.array([0.2, 0.4, 0.9, 0.7]), np.array([0, 0, 1, 1]))
    np.testing.assert_array_equal(subjects, [0, 1])
    np.testing.assert_allclose(proba, [0.3, 0.8])


def test_wilson_interval_contains_point_estimate():
    lo, hi = wilson_interval(22, 28)
    assert lo < 22 / 28 < hi
    assert 0.58 < lo < 0.61 and 0.89 < hi < 0.92


def test_binary_metrics():
    m = binary_metrics([0, 0, 1, 1], [0.1, 0.6, 0.7, 0.9], threshold=0.5)
    assert m["accuracy"] == 0.75
    assert m["sensitivity"] == 1.0 and m["specificity"] == 0.5
    assert m["confusion_matrix"] == [[1, 1], [0, 2]]


@pytest.mark.parametrize("cv", ["logo", "group_kfold"])
def test_cross_validate_never_leaks_subjects(cv):
    ds = toy_dataset()
    cfg = ExperimentConfig(evaluation=EvaluationConfig(cv=cv, n_splits=4))
    result = cross_validate(lambda: build_estimator(cfg), ds, cfg.evaluation)
    test_subjects = [s for fold in result.folds for s in fold]
    assert sorted(test_subjects) == sorted(ds.subject_ids)  # each tested exactly once
    assert not np.isnan(result.window_proba).any()
    assert result.subject_metrics["accuracy"] == 1.0  # trivially separable toy data


def test_grid_search_receives_groups():
    ds = toy_dataset(n_subjects=16)
    cfg = ExperimentConfig(
        model=ModelConfig(param_grid={"classifier__C": [0.1, 1.0]}),
        evaluation=EvaluationConfig(cv="group_kfold", n_splits=2),
    )
    result = cross_validate(lambda: build_estimator(cfg), ds, cfg.evaluation)
    assert result.subject_metrics["n"] == 16
