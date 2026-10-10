"""High-level entry points used by the CLI and notebooks."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Union

from .config import ExperimentConfig, resolve_dataset_path
from .data import WindowDataset, build_dataset, discover_recordings
from .evaluation import CVResult, cross_validate
from .models import build_estimator
from .pipeline import SchizophreniaClassifier
from .preprocessing import EEGPreprocessor
from .utils import set_seed

log = logging.getLogger(__name__)


def load_dataset(
    config: ExperimentConfig, data_path: Optional[Union[str, Path]] = None
) -> WindowDataset:
    recordings = discover_recordings(resolve_dataset_path(data_path))
    log.info("Preprocessing %d recordings", len(recordings))
    dataset = build_dataset(recordings, EEGPreprocessor(config.preprocessing))
    log.info("Dataset: %d windows of shape %s", len(dataset.y), dataset.X.shape[1:])
    return dataset


def evaluate(config: ExperimentConfig, dataset: WindowDataset) -> CVResult:
    """Subject-grouped cross-validation of the configured model."""

    def make_estimator():
        set_seed(config.seed)
        return build_estimator(config)

    return cross_validate(make_estimator, dataset, config.evaluation, seed=config.seed)


def train(
    config: ExperimentConfig,
    dataset: WindowDataset,
    output_dir: Optional[Union[str, Path]] = None,
    cv_result: Optional[CVResult] = None,
) -> SchizophreniaClassifier:
    """Fit on all subjects and optionally save, embedding any CV results."""
    classifier = SchizophreniaClassifier(config).fit(dataset)
    if cv_result is not None:
        classifier.metadata["cross_validation"] = {
            "cv": config.evaluation.cv,
            "subject_level": cv_result.subject_metrics,
            "window_level": cv_result.window_metrics,
        }
    if output_dir is not None:
        classifier.save(output_dir)
    return classifier


def write_report(result: CVResult, config: ExperimentConfig, path: Union[str, Path]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {"experiment": config.name, "config": config.to_dict(), **result.summary()}
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path
