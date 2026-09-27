"""Downloads daily OHLCV data for NSE-listed symbols from Yahoo Finance."""

from __future__ import annotations

import pandas as pd
import yfinance as yf

NSE_SUFFIX = ".NS"

REQUIRED_COLUMNS = [
    "date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
]


def to_nse_ticker(symbol: str) -> str:
    symbol = symbol.strip().upper()
    return symbol if symbol.endswith(NSE_SUFFIX) else f"{symbol}{NSE_SUFFIX}"


def download_daily_ohlcv(symbol: str, period: str = "5y") -> pd.DataFrame:
    """Downloads daily OHLCV history for a single NSE symbol.

    Returns a flat DataFrame with columns matching REQUIRED_COLUMNS,
    sorted ascending by date.
    """
    ticker = to_nse_ticker(symbol)
    raw = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if raw.empty:
        raise ValueError(f"No data returned for {ticker}")

    raw = raw.reset_index()
    # yfinance returns MultiIndex columns (field, ticker) even for a single symbol.
    raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
    raw.columns = [str(c).strip().lower().replace(" ", "_") for c in raw.columns]

    df = raw.rename(columns={"adj_close": "adjusted_close"})
    df["symbol"] = symbol.strip().upper()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

    return df[REQUIRED_COLUMNS].sort_values("date").reset_index(drop=True)


def download_universe(symbols: list[str], period: str = "5y") -> pd.DataFrame:
    """Downloads and concatenates daily OHLCV history for multiple symbols."""
    frames = [download_daily_ohlcv(symbol, period=period) for symbol in symbols]
    return pd.concat(frames, ignore_index=True)
