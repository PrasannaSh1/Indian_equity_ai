"""Data-quality checks and cleaning for daily OHLCV data."""

from __future__ import annotations

import pandas as pd

PRICE_COLUMNS = ["open", "high", "low", "close", "adjusted_close"]
REQUIRED_COLUMNS = ["date", "symbol", *PRICE_COLUMNS, "volume"]


def validate_ohlcv(df: pd.DataFrame) -> dict:
    """Reports data-quality issues without modifying the input."""
    return {
        "missing_values": int(df[REQUIRED_COLUMNS].isna().sum().sum()),
        "duplicate_rows": int(df.duplicated(subset=["date", "symbol"]).sum()),
        "non_positive_prices": int((df[PRICE_COLUMNS] <= 0).sum().sum()),
        "negative_volume": int((df["volume"] < 0).sum()),
        "high_below_low": int((df["high"] < df["low"]).sum()),
    }


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Drops rows that fail basic OHLCV sanity checks and sorts by symbol/date."""
    df = df.copy()
    df = df.dropna(subset=REQUIRED_COLUMNS)
    df = df.drop_duplicates(subset=["date", "symbol"])
    df = df[(df[PRICE_COLUMNS] > 0).all(axis=1)]
    df = df[df["volume"] >= 0]
    df = df[df["high"] >= df["low"]]
    return df.sort_values(["symbol", "date"]).reset_index(drop=True)
