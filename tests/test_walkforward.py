import pandas as pd
import pytest

from src.models.walkforward import expanding_window_folds


def test_expanding_window_folds_are_chronological_and_non_overlapping():
    dates = pd.date_range("2024-01-01", periods=20, freq="D")

    folds = expanding_window_folds(dates, n_folds=2, min_train_frac=0.5)

    assert len(folds) == 2
    (train_end_1, test_start_1, test_end_1), (train_end_2, test_start_2, test_end_2) = folds

    # fold 1: train up to day 10 (index 9), test days 11-15 (indices 10-14)
    assert train_end_1 == dates[9]
    assert test_start_1 == dates[10]
    assert test_end_1 == dates[14]

    # fold 2: train expands to include fold 1's test window; test days 16-20
    assert train_end_2 == dates[14]
    assert test_start_2 == dates[15]
    assert test_end_2 == dates[19]

    # no fold's test window starts before its own train_end
    for train_end, test_start, _ in folds:
        assert train_end < test_start

    # test windows across folds don't overlap and stay chronological
    assert test_end_1 < test_start_2


def test_expanding_window_folds_embargo_shrinks_train_end():
    dates = pd.date_range("2024-01-01", periods=20, freq="D")

    no_embargo = expanding_window_folds(dates, n_folds=2, min_train_frac=0.5, embargo_days=0)
    with_embargo = expanding_window_folds(dates, n_folds=2, min_train_frac=0.5, embargo_days=2)

    train_end_no_embargo, test_start, _ = no_embargo[0]
    train_end_with_embargo, test_start_embargo, _ = with_embargo[0]

    assert test_start == test_start_embargo  # embargo only trims train, not test
    assert train_end_with_embargo == train_end_no_embargo - pd.Timedelta(days=2)


def test_expanding_window_folds_respects_min_train_frac():
    dates = pd.date_range("2024-01-01", periods=10, freq="D")

    folds = expanding_window_folds(dates, n_folds=1, min_train_frac=0.7)

    train_end, test_start, test_end = folds[0]
    assert train_end == dates[6]  # 70% of 10 = 7 -> first 7 dates (indices 0-6) are train-only
    assert test_start == dates[7]
    assert test_end == dates[9]
