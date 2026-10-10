"""Locate recordings on disk and assemble them into a windowed dataset."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union

import mne
import numpy as np

from .preprocessing import EEGPreprocessor

HEALTHY, SCHIZOPHRENIA = 0, 1
CLASS_NAMES = ("healthy", "schizophrenia")

# Files are named h01.edf ... h14.edf (controls) and s01.edf ... s14.edf (patients).
_FILENAME = re.compile(r"^(?P<group>[hs])(?P<num>\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class Recording:
    subject_id: str
    label: int
    path: Path


@dataclass
class WindowDataset:
    """Windows from many subjects.

    ``X`` is ``(n_windows, n_channels, n_times)``; ``y`` and ``groups`` hold the
    label and subject index of each window, and ``subject_ids[groups[i]]`` names
    the subject window ``i`` came from.
    """

    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    subject_ids: List[str]

    @property
    def subject_labels(self) -> np.ndarray:
        labels = np.empty(len(self.subject_ids), dtype=int)
        labels[self.groups] = self.y
        return labels


def parse_recording(path: Union[str, Path]) -> Optional[Recording]:
    path = Path(path)
    match = _FILENAME.match(path.stem)
    if match is None:
        return None
    label = HEALTHY if match["group"].lower() == "h" else SCHIZOPHRENIA
    return Recording(subject_id=path.stem.lower(), label=label, path=path)


def discover_recordings(dataset_dir: Union[str, Path]) -> List[Recording]:
    recordings = [
        rec
        for path in sorted(Path(dataset_dir).glob("*.edf"))
        if (rec := parse_recording(path)) is not None
    ]
    if not recordings:
        raise FileNotFoundError(f"No h*/s* .edf recordings found in {dataset_dir}")
    return recordings


def read_raw(path: Union[str, Path]) -> mne.io.BaseRaw:
    return mne.io.read_raw_edf(path, preload=True, verbose="ERROR")


def build_dataset(
    recordings: Sequence[Recording], preprocessor: EEGPreprocessor
) -> WindowDataset:
    X, y, groups = [], [], []
    for index, rec in enumerate(recordings):
        windows = preprocessor.transform(read_raw(rec.path))
        X.append(windows)
        y.append(np.full(len(windows), rec.label, dtype=int))
        groups.append(np.full(len(windows), index, dtype=int))
    return WindowDataset(
        X=np.concatenate(X),
        y=np.concatenate(y),
        groups=np.concatenate(groups),
        subject_ids=[rec.subject_id for rec in recordings],
    )
