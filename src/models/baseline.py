"""Naive baseline: predicts the training-set base rate for every sample.

This is the simplest defensible baseline for a probabilistic classifier -- it
encodes no information beyond "how often did the market go up historically" --
and by construction should score ROC-AUC == 0.5 (no discriminative power),
which doubles as a sanity check on the evaluation pipeline itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def naive_base_rate(y_train: pd.Series) -> float:
    return float(y_train.mean())


def naive_predict_proba(y_train: pd.Series, n: int) -> np.ndarray:
    return np.full(n, naive_base_rate(y_train))


def naive_predict(y_train: pd.Series, n: int) -> np.ndarray:
    base_rate = naive_base_rate(y_train)
    label = 1 if base_rate > 0.5 else 0
    return np.full(n, label)
