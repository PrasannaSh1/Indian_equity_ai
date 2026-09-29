import pandas as pd
import pytest

from src.explainability.confidence import confidence_label, prediction_confidence


def test_prediction_confidence_matches_hand_computed_distance_from_coin_flip():
    proba = pd.Series([0.5, 0.75, 0.9, 0.1])
    out = prediction_confidence(proba)

    assert out.tolist() == pytest.approx([0.0, 0.5, 0.8, 0.8])


def test_confidence_label_buckets_correctly():
    confidence = pd.Series([0.1, 0.35, 0.7])
    labels = confidence_label(confidence)
    assert labels.tolist() == ["Low", "Medium", "High"]
