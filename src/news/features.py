"""Aggregates per-article sentiment into a daily (date, symbol) feature table,
joinable onto the price/technical feature table -- see project plan Section 23
(NEWS block of the multimodal feature vector).
"""

from __future__ import annotations

import pandas as pd


def aggregate_daily_news_features(news_df: pd.DataFrame) -> pd.DataFrame:
    """Requires columns: symbol, published_timestamp, positive_probability,
    negative_probability. Groups by the calendar date of publish (UTC), per symbol.
    """
    df = news_df.copy()
    df["date"] = pd.to_datetime(df["published_timestamp"]).dt.tz_localize(None).dt.normalize()
    df["net_sentiment"] = df["positive_probability"] - df["negative_probability"]

    grouped = df.groupby(["date", "symbol"])
    return grouped.agg(
        news_count=("net_sentiment", "count"),
        sentiment_mean=("net_sentiment", "mean"),
        sentiment_std=("net_sentiment", "std"),
        positive_news_ratio=("positive_probability", lambda s: (s > 0.5).mean()),
        negative_news_ratio=("negative_probability", lambda s: (s > 0.5).mean()),
    ).reset_index()
