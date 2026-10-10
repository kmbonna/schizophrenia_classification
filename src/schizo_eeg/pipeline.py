"""End-to-end classifier: recording in, subject-level prediction out.

A ``SchizophreniaClassifier`` bundles the preprocessing config with a fitted
window estimator (features + scaler + model). Saving it writes everything
needed to reproduce a prediction, so a new recording is transformed exactly
as the training data was::

    clf = SchizophreniaClassifier.load("artifacts/logreg_bandpower")
    clf.predict_file("new_subject.edf")
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import platform
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import mne
import numpy as np

from . import __version__
from .config import ExperimentConfig
from .data import CLASS_NAMES, WindowDataset, read_raw
from .models import build_estimator
from .preprocessing import EEGPreprocessor
from .utils import fit_with_groups, set_seed

log = logging.getLogger(__name__)

MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"


@dataclass
class Prediction:
    probability: float  # P(schizophrenia), mean over windows
    label: int
    label_name: str
    threshold: float
    n_windows: int
    window_probabilities: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _library_versions() -> Dict[str, str]:
    import scipy
    import sklearn

    versions = {
        "schizo_eeg": __version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "mne": mne.__version__,
    }
    try:
        import sys

        if "tensorflow" in sys.modules:
            versions["tensorflow"] = sys.modules["tensorflow"].__version__
    except Exception:  # pragma: no cover
        pass
    return versions


class SchizophreniaClassifier:
    def __init__(self, config: ExperimentConfig, estimator=None, metadata: Optional[dict] = None):
        self.config = config
        self.preprocessor = EEGPreprocessor(config.preprocessing)
        self.estimator = estimator
        self.metadata = metadata or {}

    # --- training ----------------------------------------------------------

    def fit(self, dataset: WindowDataset) -> "SchizophreniaClassifier":
        set_seed(self.config.seed)
        self.estimator = build_estimator(self.config)
        fit_with_groups(self.estimator, dataset.X, dataset.y, dataset.groups)
        self.metadata.update(
            {
                "trained_at": dt.datetime.now().isoformat(timespec="seconds"),
                "training_subjects": list(dataset.subject_ids),
                "n_training_windows": int(len(dataset.y)),
            }
        )
        return self

    # --- inference ---------------------------------------------------------

    def _check_fitted(self):
        if self.estimator is None:
            raise RuntimeError("Classifier is not fitted; call fit() or load().")

    def predict_windows(self, windows: np.ndarray) -> np.ndarray:
        """P(schizophrenia) for each preprocessed window."""
        self._check_fitted()
        return self.estimator.predict_proba(windows)[:, 1]

    def predict_raw(self, raw: mne.io.BaseRaw) -> Prediction:
        windows = self.preprocessor.transform(raw)
        window_proba = self.predict_windows(windows)
        probability = float(window_proba.mean())
        threshold = self.config.evaluation.threshold
        label = int(probability > threshold)
        return Prediction(
            probability=probability,
            label=label,
            label_name=CLASS_NAMES[label],
            threshold=threshold,
            n_windows=int(len(window_proba)),
            window_probabilities=[float(p) for p in window_proba],
        )

    def predict_file(self, path: Union[str, Path]) -> Prediction:
        return self.predict_raw(read_raw(path))

    # --- persistence -------------------------------------------------------

    def save(self, directory: Union[str, Path]) -> Path:
        self._check_fitted()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.estimator, directory / MODEL_FILE)
        metadata = {
            **self.metadata,
            "config": self.config.to_dict(),
            "versions": _library_versions(),
        }
        (directory / METADATA_FILE).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        log.info("Saved model to %s", directory)
        return directory

    @classmethod
    def load(cls, directory: Union[str, Path]) -> "SchizophreniaClassifier":
        directory = Path(directory)
        metadata = json.loads((directory / METADATA_FILE).read_text(encoding="utf-8"))
        config = ExperimentConfig.from_dict(metadata["config"])

        if config.model.name in ("cnn1d", "chrononet", "cnn_gru"):
            import schizo_eeg.models.deep  # noqa: F401  (registers TensorFlow)

        saved = metadata.get("versions", {})
        current = _library_versions()
        for lib in ("scikit-learn", "tensorflow"):
            if lib in saved and lib in current and saved[lib] != current[lib]:
                warnings.warn(
                    f"Model was saved with {lib} {saved[lib]} but {current[lib]} is "
                    "installed; predictions may differ.",
                    RuntimeWarning,
                )
        estimator = joblib.load(directory / MODEL_FILE)
        return cls(config, estimator, metadata)
