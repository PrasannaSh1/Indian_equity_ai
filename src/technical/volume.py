"""Volume indicators: Volume MA, Volume Ratio, OBV, VWAP."""

from __future__ import annotations

import numpy as np
import pandas as pd


def volume_ma(volume: pd.Series, window: int = 20) -> pd.Series:
    return volume.rolling(window=window).mean()


def volume_ratio(volume: pd.Series, window: int = 20) -> pd.Series:
    return volume / volume_ma(volume, window)


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume: cumulative signed volume based on the daily close direction."""
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def vwap(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 20
) -> pd.Series:
    """Rolling volume-weighted average price over `window` bars, using typical price."""
    typical_price = (high + low + close) / 3
    pv = typical_price * volume
    return pv.rolling(window=window).sum() / volume.rolling(window=window).sum()
