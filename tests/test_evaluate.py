import numpy as np
import pytest

from src.models.evaluate import classification_metrics


def test_classification_metrics_on_a_hand_computed_confusion_matrix():
    # 4 samples: TP, TN, FP, FN
    y_true = np.array([1, 0, 0, 1])
    y_pred = np.array([1, 0, 1, 0])
    y_proba = np.array([0.9, 0.1, 0.6, 0.4])

    out = classification_metrics(y_true, y_pred, y_proba)

    assert out["accuracy"] == pytest.approx(0.5)
    assert out["precision"] == pytest.approx(0.5)  # 1 TP / (1 TP + 1 FP)
    assert out["recall"] == pytest.approx(0.5)  # 1 TP / (1 TP + 1 FN)
    assert out["f1"] == pytest.approx(0.5)


def test_classification_metrics_perfect_predictions():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0])
    y_proba = np.array([0.99, 0.01, 0.99, 0.01])

    out = classification_metrics(y_true, y_pred, y_proba)

    assert out["accuracy"] == 1.0
    assert out["precision"] == 1.0
    assert out["recall"] == 1.0
    assert out["roc_auc"] == 1.0
    assert out["brier_score"] < 0.01


def test_classification_metrics_roc_auc_is_nan_for_a_single_class_split():
    y_true = np.array([1, 1, 1])
    y_pred = np.array([1, 1, 1])
    y_proba = np.array([0.8, 0.9, 0.7])

    out = classification_metrics(y_true, y_pred, y_proba)

    assert np.isnan(out["roc_auc"])
