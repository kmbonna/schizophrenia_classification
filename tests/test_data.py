from pathlib import Path

import numpy as np

from conftest import make_raw
from schizo_eeg.config import PreprocessingConfig
from schizo_eeg import data
from schizo_eeg.preprocessing import EEGPreprocessor


def test_parse_recording_labels_from_filename():
    assert data.parse_recording("x/h03.edf").label == data.HEALTHY
    assert data.parse_recording("x/S11.edf").label == data.SCHIZOPHRENIA
    assert data.parse_recording("x/MANIFEST.edf") is None
    # Folder names must not influence the label (the old notebook checked
    # for 'h'/'s' anywhere in the path).
    assert data.parse_recording("schizo/healthy/h01.edf").subject_id == "h01"


def test_build_dataset(monkeypatch):
    recordings = [
        data.Recording("h01", data.HEALTHY, Path("h01.edf")),
        data.Recording("s01", data.SCHIZOPHRENIA, Path("s01.edf")),
    ]
    lengths = {"h01.edf": 30.0, "s01.edf": 22.0}
    monkeypatch.setattr(data, "read_raw", lambda p: make_raw(seconds=lengths[str(p)]))

    ds = data.build_dataset(recordings, EEGPreprocessor(PreprocessingConfig()))

    assert ds.X.shape[0] == ds.y.shape[0] == ds.groups.shape[0] == 7 + 5
    np.testing.assert_array_equal(ds.groups, [0] * 7 + [1] * 5)
    np.testing.assert_array_equal(ds.y, [0] * 7 + [1] * 5)
    np.testing.assert_array_equal(ds.subject_labels, [0, 1])
