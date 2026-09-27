"""Composite 0-100 Technical Score."""

from __future__ import annotations

import pandas as pd


def technical_score(df: pd.DataFrame) -> pd.Series:
    """Averages three equally weighted 0-100 sub-scores, requiring columns from
    add_technical_indicators: sma_50, sma_200, rsi_14, volume_ratio.

    - trend: 100 if close > sma_50 > sma_200, 50 if only one holds, 0 if neither
    - momentum: RSI(14), used directly since it is already a 0-100 oscillator
    - volume confirmation: volume_ratio scaled so 1.0x average volume = 50, clipped to [0, 100]
    """
    required = df[["close", "sma_50", "sma_200", "rsi_14", "volume_ratio"]]
    missing_inputs = required.isna().any(axis=1)

    trend = (df["close"] > df["sma_50"]).astype(float) * 50 + (
        df["sma_50"] > df["sma_200"]
    ).astype(float) * 50
    momentum = df["rsi_14"]
    volume_component = (df["volume_ratio"] * 50).clip(lower=0, upper=100)

    score = (trend + momentum + volume_component) / 3
    score = score.clip(lower=0, upper=100)
    return score.where(~missing_inputs)
