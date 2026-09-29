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
    dates: pd.Series, n_folds: int = 4, min_train_frac: float = 0.5, embargo_days: int = 0
) -> list[tuple[np.datetime64, np.datetime64, np.datetime64]]:
    """Returns `n_folds` (train_end, test_start, test_end) cutoffs.

    The first `min_train_frac` of the date range is reserved as the minimum
    training window; the remaining region is split into `n_folds` equal,
    non-overlapping, chronologically ordered test windows, each fold's training
    set expanding to include everything before its test window.

    `embargo_days` drops the last `embargo_days` unique dates before test_start
    from the training window. This matters because next_day_direction/return
    labels are built from the *following* day's price -- the very last training
    row before test_start would otherwise be labeled using a price that falls
    inside (or at the boundary of) the test window. An embargo of >=1 fully
    removes that overlap for a 1-day-ahead target; purged CV for longer-horizon
    targets would need a larger embargo, proportional to the label horizon.
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
        train_end_idx = test_region_start_idx + boundary[0] - 1 - embargo_days
        if train_end_idx < 0:
            continue  # embargo would consume the entire training window
        train_end = unique_dates[train_end_idx]
        folds.append((train_end, test_start, test_end))
    return folds
