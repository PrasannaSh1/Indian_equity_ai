"""Classification metrics for next-day-direction models."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    """accuracy, precision, recall, F1, ROC-AUC, log loss, Brier score.

    ROC-AUC is undefined (returned as NaN) when y_true contains only one class,
    e.g. a tiny or degenerate evaluation split.
    """
    roc_auc = roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else float("nan")
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc,
        "log_loss": log_loss(y_true, y_proba, labels=[0, 1]),
        "brier_score": brier_score_loss(y_true, y_proba),
    }
