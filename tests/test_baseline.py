import pandas as pd
import pytest

from src.models.baseline import naive_base_rate, naive_predict, naive_predict_proba


def test_naive_base_rate_matches_hand_computed_mean():
    y_train = pd.Series([1, 1, 1, 0])
    assert naive_base_rate(y_train) == pytest.approx(0.75)


def test_naive_predict_proba_is_constant_base_rate_for_every_row():
    y_train = pd.Series([1, 0, 1, 0, 1])
    out = naive_predict_proba(y_train, n=3)

    assert out.tolist() == pytest.approx([0.6, 0.6, 0.6])


def test_naive_predict_uses_majority_class():
    up_majority = pd.Series([1, 1, 0])
    down_majority = pd.Series([0, 0, 1])

    assert naive_predict(up_majority, n=2).tolist() == [1, 1]
    assert naive_predict(down_majority, n=2).tolist() == [0, 0]
