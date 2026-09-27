"""Return and rolling-volatility feature calculations."""

from __future__ import annotations

import pandas as pd

RETURN_PERIODS = (1, 5, 20, 60)
VOLATILITY_WINDOWS = (10, 20, 30, 60)


def add_returns(df: pd.DataFrame, periods: tuple[int, ...] = RETURN_PERIODS) -> pd.DataFrame:
    """Adds return_{p}d = close.pct_change(p), computed per symbol."""
    df = df.sort_values(["symbol", "date"]).copy()
    for p in periods:
        df[f"return_{p}d"] = df.groupby("symbol")["close"].pct_change(p)
    return df


def add_rolling_volatility(
    df: pd.DataFrame, windows: tuple[int, ...] = VOLATILITY_WINDOWS
) -> pd.DataFrame:
    """Adds volatility_{w}d = rolling std of the 1-day return, per symbol."""
    df = df.sort_values(["symbol", "date"]).copy()
    if "return_1d" not in df.columns:
        df["return_1d"] = df.groupby("symbol")["close"].pct_change(1)
    for w in windows:
        df[f"volatility_{w}d"] = df.groupby("symbol")["return_1d"].transform(
            lambda s, w=w: s.rolling(window=w).std()
        )
    return df
