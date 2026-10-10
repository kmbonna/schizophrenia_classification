import json

from conftest import make_raw
from schizo_eeg import cli, pipeline, training
from schizo_eeg.config import ExperimentConfig, ModelConfig
from test_pipeline import synthetic_dataset


def test_train_then_predict(tmp_path, monkeypatch, capsys):
    config = ExperimentConfig(name="tiny", model=ModelConfig(features=["bandpower"]))
    config_path = tmp_path / "tiny.yaml"
    config.to_yaml(config_path)

    monkeypatch.setattr(training, "load_dataset", lambda cfg, data=None: synthetic_dataset(cfg))
    out = tmp_path / "artifact"
    assert cli.main(["train", "--config", str(config_path), "--out", str(out), "--evaluate"]) == 0
    metadata = json.loads((out / "metadata.json").read_text())
    assert "cross_validation" in metadata

    monkeypatch.setattr(pipeline, "read_raw", lambda p: make_raw(seconds=30, alpha_amp=40e-6))
    capsys.readouterr()
    assert cli.main(["predict", "--model", str(out), "--json", "patient.edf"]) == 0
    [result] = json.loads(capsys.readouterr().out)
    assert result["label_name"] == "schizophrenia"
    assert "window_probabilities" not in result
