"""Expanding-window walk-forward validation folds.

A lightweight walk-forward scheme for comparing feature sets/models (used in
Phase 7 to test hypotheses H1-H5). This is intentionally simpler than the
dedicated backtesting engine built in Phase 9 (no transaction costs, position
sizing, or entry/exit logic here) -- it only answers "does this feature set
generalize across multiple out-of-sample periods," not "is this profitable to
trade."
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def expanding_window_folds(
    dates: pd.Series, n_folds: int = 4, min_train_frac: float = 0.5
) -> list[tuple[np.datetime64, np.datetime64, np.datetime64]]:
    """Returns `n_folds` (train_end, test_start, test_end) cutoffs.

    The first `min_train_frac` of the date range is reserved as the minimum
    training window; the remaining region is split into `n_folds` equal,
    non-overlapping, chronologically ordered test windows, each fold's training
    set expanding to include everything before its test window.
    """
    unique_dates = np.sort(pd.Series(dates).unique())
    n = len(unique_dates)
    test_region_start_idx = int(n * min_train_frac)
    test_region = unique_dates[test_region_start_idx:]

    fold_boundaries = np.array_split(np.arange(len(test_region)), n_folds)

    folds = []
    for boundary in fold_boundaries:
        if len(boundary) == 0:
            continue
        test_start = test_region[boundary[0]]
        test_end = test_region[boundary[-1]]
        train_end = unique_dates[test_region_start_idx + boundary[0] - 1]
        folds.append((train_end, test_start, test_end))
    return folds
