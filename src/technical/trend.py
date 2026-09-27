"""Trend indicators: SMA, EMA."""

from __future__ import annotations

import pandas as pd

SMA_WINDOWS = (20, 50, 100, 200)
EMA_SPANS = (20, 50)


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()
