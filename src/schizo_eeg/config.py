"""Experiment configuration.

Every setting that affects how a recording becomes a prediction lives here, so
that it can be saved next to a trained model and replayed at inference time.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

# Standard 10-20 order. The EDF files store channels in a different order, so
# channels are always picked by name in this order rather than by position.
CHANNELS: List[str] = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T3", "C3", "Cz", "C4", "T4",
    "T5", "P3", "Pz", "P4", "T6",
    "O1", "O2",
]


@dataclass
class PreprocessingConfig:
    """How a continuous recording is turned into fixed-length windows."""

    channels: List[str] = field(default_factory=lambda: list(CHANNELS))
    sfreq: float = 250.0
    reference: str = "average"
    l_freq: float = 0.5
    h_freq: float = 45.0
    epoch_duration: float = 5.0
    epoch_overlap: float = 1.0

    @property
    def n_times(self) -> int:
        return int(round(self.epoch_duration * self.sfreq))


@dataclass
class ModelConfig:
    """Which estimator to build and its keyword arguments."""

    name: str = "logistic_regression"
    params: Dict[str, Any] = field(default_factory=dict)
    # Feature families for classical models ("time", "bandpower"); ignored by
    # deep models, which learn from the raw windows.
    features: List[str] = field(default_factory=lambda: ["time"])
    # Optional hyperparameter grid, tuned with an inner subject-grouped CV.
    param_grid: Optional[Dict[str, List[Any]]] = None


@dataclass
class EvaluationConfig:
    cv: str = "logo"  # "logo" or "group_kfold"
    n_splits: int = 5
    # A subject is predicted schizophrenic when the mean window probability
    # exceeds this. Fixed in advance, never tuned on the evaluation folds.
    threshold: float = 0.5


@dataclass
class ExperimentConfig:
    name: str = "experiment"
    seed: int = 42
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        data = dict(data)
        return cls(
            name=data.get("name", "experiment"),
            seed=data.get("seed", 42),
            preprocessing=PreprocessingConfig(**data.get("preprocessing", {})),
            model=ModelConfig(**data.get("model", {})),
            evaluation=EvaluationConfig(**data.get("evaluation", {})),
        )

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ExperimentConfig":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    def to_yaml(self, path: Union[str, Path]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)


def resolve_dataset_path(path: Optional[Union[str, Path]] = None) -> Path:
    """Return the dataset folder from an explicit path or $DATASET_PATH (.env)."""
    if path is None:
        from dotenv import load_dotenv

        load_dotenv()
        path = os.getenv("DATASET_PATH")
    if not path:
        raise ValueError(
            "No dataset path given. Pass --data or set DATASET_PATH in a .env file."
        )
    resolved = Path(path).expanduser()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {resolved}")
    return resolved
