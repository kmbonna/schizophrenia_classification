import json

import numpy as np
import pytest

from conftest import make_raw
from schizo_eeg.config import ExperimentConfig, ModelConfig
from schizo_eeg.data import WindowDataset
from schizo_eeg.pipeline import METADATA_FILE, SchizophreniaClassifier


def synthetic_dataset(config, n_subjects=6):
    """'Patients' carry a strong 10 Hz rhythm so the model has something to learn."""
    pre = SchizophreniaClassifier(config).preprocessor
    X, y, groups = [], [], []
    for s in range(n_subjects):
        label = s % 2
        w = pre.transform(make_raw(seconds=30, seed=s, alpha_amp=40e-6 * label))
        X.append(w)
        y += [label] * len(w)
        groups += [s] * len(w)
    return WindowDataset(np.concatenate(X), np.array(y), np.array(groups),
                         [f"sub{s}" for s in range(n_subjects)])


@pytest.fixture
def config():
    return ExperimentConfig(model=ModelConfig(name="logistic_regression", features=["bandpower"]))


def test_predict_requires_fit(config):
    with pytest.raises(RuntimeError, match="not fitted"):
        SchizophreniaClassifier(config).predict_raw(make_raw())


def test_end_to_end_save_load_predict(config, tmp_path):
    clf = SchizophreniaClassifier(config).fit(synthetic_dataset(config))

    healthy = make_raw(seconds=30, seed=100)
    patient = make_raw(seconds=30, seed=101, alpha_amp=40e-6)
    assert clf.predict_raw(healthy).label_name == "healthy"
    assert clf.predict_raw(patient).label_name == "schizophrenia"

    clf.save(tmp_path / "model")
    metadata = json.loads((tmp_path / "model" / METADATA_FILE).read_text())
    assert metadata["config"]["model"]["features"] == ["bandpower"]
    assert metadata["training_subjects"][0] == "sub0"
    assert "scikit-learn" in metadata["versions"]

    loaded = SchizophreniaClassifier.load(tmp_path / "model")
    original, restored = clf.predict_raw(patient), loaded.predict_raw(patient)
    assert restored.probability == pytest.approx(original.probability)
    assert restored.n_windows == 7


def test_loaded_model_applies_saved_preprocessing(tmp_path):
    """A model trained on 10 s windows must window new data at 10 s too."""
    config = ExperimentConfig(model=ModelConfig(features=["bandpower"]))
    config.preprocessing.epoch_duration = 10.0
    config.preprocessing.epoch_overlap = 0.0
    SchizophreniaClassifier(config).fit(synthetic_dataset(config)).save(tmp_path / "m")

    prediction = SchizophreniaClassifier.load(tmp_path / "m").predict_raw(make_raw(seconds=30))
    assert prediction.n_windows == 3
