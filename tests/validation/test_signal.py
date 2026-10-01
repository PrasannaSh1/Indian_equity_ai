import pytest

from src.validation.signal import classify_signal


def test_classify_signal_downgrades_to_low_confidence_regardless_of_direction():
    # The audit's own reported example: probability slightly above 50% but very low confidence.
    result = classify_signal(probability_up=0.579, confidence=0.11)

    assert result.label == "Low Confidence"


def test_classify_signal_labels_strong_bullish_at_high_confidence():
    result = classify_signal(probability_up=0.85, confidence=0.7)

    assert result.label == "Strong Bullish"


def test_classify_signal_labels_moderate_bullish_at_medium_confidence():
    result = classify_signal(probability_up=0.65, confidence=0.3)

    assert result.label == "Bullish"


def test_classify_signal_labels_strong_bearish_at_high_confidence():
    result = classify_signal(probability_up=0.1, confidence=0.8)

    assert result.label == "Strong Bearish"


def test_classify_signal_labels_moderate_bearish_at_medium_confidence():
    result = classify_signal(probability_up=0.3, confidence=0.4)

    assert result.label == "Bearish"


@pytest.mark.parametrize("probability_up", [0.45, 0.5, 0.55])
def test_classify_signal_near_coin_flip_is_low_confidence(probability_up):
    result = classify_signal(probability_up=probability_up, confidence=abs(probability_up - 0.5) * 2)

    assert result.label == "Low Confidence"
