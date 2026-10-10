"""Feature-based scikit-learn classifiers."""

from __future__ import annotations

from typing import Any, Dict, Sequence

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ..features import build_feature_extractor

CLASSIFIERS = {
    "logistic_regression": lambda seed, **kw: LogisticRegression(
        max_iter=kw.pop("max_iter", 2000), random_state=seed, **kw
    ),
    "random_forest": lambda seed, **kw: RandomForestClassifier(
        n_jobs=kw.pop("n_jobs", -1), random_state=seed, **kw
    ),
    "svm": lambda seed, **kw: SVC(probability=True, random_state=seed, **kw),
}


def build_classical_pipeline(
    name: str,
    params: Dict[str, Any],
    features: Sequence[str],
    sfreq: float,
    seed: int,
) -> Pipeline:
    """windows -> features -> standardise -> classifier."""
    return Pipeline(
        [
            ("features", build_feature_extractor(features, sfreq=sfreq)),
            ("scaler", StandardScaler()),
            ("classifier", CLASSIFIERS[name](seed, **dict(params))),
        ]
    )
