"""Builds the next-day-direction ML dataset from Phase 1-3 features.

Combines Phase 3's technical indicators with Phase 1's return/volatility
features, normalizes price-level indicators (SMA/EMA/MACD/ATR are in absolute
rupee terms and not comparable across stocks of very different price scales)
into ratios, and labels each row with whether the *next* trading day closed
higher than the current day.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.returns import add_returns, add_rolling_volatility

FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "return_20d",
    "return_60d",
    "volatility_10d",
    "volatility_20d",
    "volatility_30d",
    "volatility_60d",
    "price_to_sma20",
    "price_to_sma50",
    "price_to_sma200",
    "sma50_to_sma200",
    "ema20_to_ema50",
    "rsi_14",
    "macd_norm",
    "macd_histogram_norm",
    "roc_12",
    "stoch_k",
    "stoch_d",
    "adx_14",
    "atr_pct",
    "bb_width",
    "volume_ratio",
    "technical_score",
]

TARGET_COLUMN = "next_day_direction"

# Bumped whenever FEATURE_COLUMNS or their computation changes in a way that would
# make an older persisted model's inputs incompatible with the current pipeline
# (Phase 19/32). A trained model's registry entry (src.models.registry) records the
# schema version it was trained against; src.models.predict refuses to serve a model
# whose recorded version doesn't match this constant, rather than silently feeding it
# a differently-shaped feature vector.
FEATURE_SCHEMA_VERSION = "1.0"


def build_feature_table(technical_df: pd.DataFrame) -> pd.DataFrame:
    """Adds return/volatility features and normalizes absolute-price indicators to ratios."""
    df = technical_df.sort_values(["symbol", "date"]).copy()
    df = add_returns(df)
    df = add_rolling_volatility(df)

    df["price_to_sma20"] = df["close"] / df["sma_20"] - 1
    df["price_to_sma50"] = df["close"] / df["sma_50"] - 1
    df["price_to_sma200"] = df["close"] / df["sma_200"] - 1
    df["sma50_to_sma200"] = df["sma_50"] / df["sma_200"] - 1
    df["ema20_to_ema50"] = df["ema_20"] / df["ema_50"] - 1
    df["macd_norm"] = df["macd"] / df["close"]
    df["macd_histogram_norm"] = df["macd_histogram"] / df["close"]
    df["atr_pct"] = df["atr_14"] / df["close"]

    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Adds next_day_direction = 1 if tomorrow's close > today's close, else 0.

    The last row of each symbol has no "tomorrow" -- its target is left as NaN
    rather than silently defaulting to 0 (a naive `(next_close > close)` cast
    would treat the NaN comparison, which pandas evaluates as False, as a real
    "down" label).
    """
    df = df.sort_values(["symbol", "date"]).copy()
    next_close = df.groupby("symbol")["close"].shift(-1)
    direction = (next_close > df["close"]).astype(float)
    df[TARGET_COLUMN] = direction.where(next_close.notna())
    return df


def add_next_day_return(df: pd.DataFrame) -> pd.DataFrame:
    """Adds next_day_return: tomorrow's close-to-close percent return, continuous
    (the regression counterpart of next_day_direction). The last row of each
    symbol has no "tomorrow" and is left as NaN via shift(-1), which already
    produces NaN rather than a fabricated value -- no comparison-with-NaN pitfall
    here since this is arithmetic, not a boolean cast.
    """
    df = df.sort_values(["symbol", "date"]).copy()
    next_close = df.groupby("symbol")["close"].shift(-1)
    df["next_day_return"] = next_close / df["close"] - 1
    return df


def build_ml_dataset(technical_df: pd.DataFrame) -> pd.DataFrame:
    """Full Phase 5 dataset: features + target, with warm-up/undefined rows dropped."""
    df = build_feature_table(technical_df)
    df = add_target(df)
    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    return df.dropna(subset=required).reset_index(drop=True)


def build_latest_features(technical_df: pd.DataFrame) -> pd.DataFrame:
    """Builds the same feature table as build_feature_table/build_ml_dataset -- the
    identical transformation, so training and live inference can never silently
    diverge (Phase 19's "same feature schema" requirement) -- but keeps only the
    latest row per symbol and does not require a target.

    build_ml_dataset can't be reused directly for this: it calls add_target, which
    needs *tomorrow's* close to label today's row, so it always drops the most
    recent day (the one row live inference actually needs a prediction for) as NaN.
    """
    df = build_feature_table(technical_df)
    return df.sort_values(["symbol", "date"]).groupby("symbol").tail(1).reset_index(drop=True)


def chronological_split(
    df: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits by global calendar-date cutoffs (not per-symbol), so validation and
    test always cover calendar periods strictly after training -- no leakage.
    """
    unique_dates = np.sort(df["date"].unique())
    n = len(unique_dates)
    train_end = unique_dates[int(n * train_frac) - 1]
    val_end = unique_dates[int(n * (train_frac + val_frac)) - 1]

    train = df[df["date"] <= train_end]
    val = df[(df["date"] > train_end) & (df["date"] <= val_end)]
    test = df[df["date"] > val_end]
    return train, val, test
