"""Applies the full technical-indicator set per symbol to a multi-stock daily OHLCV table."""

from __future__ import annotations

import pandas as pd

from src.technical.momentum import adx, macd, roc, rsi, stochastic_oscillator
from src.technical.trend import EMA_SPANS, SMA_WINDOWS, ema, sma
from src.technical.volatility import atr, bollinger_bands
from src.technical.volume import obv, vwap, volume_ma, volume_ratio


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Adds trend, momentum, volatility, and volume indicators, computed independently per symbol."""
    frames = []
    for _, group in df.sort_values("date").groupby("symbol"):
        g = group.copy()

        for w in SMA_WINDOWS:
            g[f"sma_{w}"] = sma(g["close"], w)
        for span in EMA_SPANS:
            g[f"ema_{span}"] = ema(g["close"], span)

        g["rsi_14"] = rsi(g["close"], period=14)
        macd_df = macd(g["close"])
        g["macd"] = macd_df["macd"]
        g["macd_signal"] = macd_df["signal"]
        g["macd_histogram"] = macd_df["histogram"]
        g["roc_12"] = roc(g["close"], period=12)
        stoch_df = stochastic_oscillator(g["high"], g["low"], g["close"])
        g["stoch_k"] = stoch_df["percent_k"]
        g["stoch_d"] = stoch_df["percent_d"]
        g["adx_14"] = adx(g["high"], g["low"], g["close"], period=14)

        g["atr_14"] = atr(g["high"], g["low"], g["close"], period=14)
        bb_df = bollinger_bands(g["close"])
        g["bb_middle"] = bb_df["bb_middle"]
        g["bb_upper"] = bb_df["bb_upper"]
        g["bb_lower"] = bb_df["bb_lower"]
        g["bb_width"] = bb_df["bb_width"]

        g["volume_ma_20"] = volume_ma(g["volume"], window=20)
        g["volume_ratio"] = volume_ratio(g["volume"], window=20)
        g["obv"] = obv(g["close"], g["volume"])
        g["vwap_20"] = vwap(g["high"], g["low"], g["close"], g["volume"], window=20)

        frames.append(g)

    return pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
