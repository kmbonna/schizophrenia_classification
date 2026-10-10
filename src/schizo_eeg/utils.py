from __future__ import annotations

import inspect
import os
import random
import sys

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and (if already imported) TensorFlow."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if "tensorflow" in sys.modules:
        import tensorflow as tf

        tf.random.set_seed(seed)


def fit_with_groups(estimator, X, y, groups):
    """Call ``fit``, forwarding ``groups`` when the estimator accepts it."""
    if "groups" in inspect.signature(estimator.fit).parameters:
        return estimator.fit(X, y, groups=groups)
    return estimator.fit(X, y)
