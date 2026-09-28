"""Historical event-impact analysis: how has a given event type historically
moved the stock, at several forward horizons (project plan Section 15)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def explode_event_types(df: pd.DataFrame) -> pd.DataFrame:
    """Expands the `event_types` list column into one row per (news item, event_type)."""
    exploded = df.explode("event_types").dropna(subset=["event_types"])
    return exploded.rename(columns={"event_types": "event_type"}).reset_index(drop=True)


def _forward_returns(event_date: pd.Timestamp, symbol_prices: pd.DataFrame, horizons) -> dict:
    """Forward return at each horizon from the first trading day on/after event_date."""
    dates = symbol_prices["date"].to_numpy()
    idx = int(np.searchsorted(dates, np.datetime64(event_date)))
    if idx >= len(dates):
        return {h: np.nan for h in horizons}

    base_close = symbol_prices["close"].iloc[idx]
    result = {}
    for h in horizons:
        target_idx = idx + h
        if target_idx < len(symbol_prices):
            result[h] = symbol_prices["close"].iloc[target_idx] / base_close - 1
        else:
            result[h] = np.nan
    return result


def compute_event_occurrence_returns(
    events: pd.DataFrame, prices: pd.DataFrame, horizons: tuple[int, ...] = (1, 5, 20)
) -> pd.DataFrame:
    """One row per (event occurrence), with the forward return at each horizon.

    `events` must have columns: symbol, event_type, published_timestamp.
    `prices` must have columns: symbol, date, close.
    """
    rows = []
    for _, event in events.iterrows():
        symbol_prices = (
            prices[prices["symbol"] == event["symbol"]].sort_values("date").reset_index(drop=True)
        )
        if symbol_prices.empty:
            continue
        event_date = pd.Timestamp(event["published_timestamp"]).tz_localize(None).normalize()
        forward = _forward_returns(event_date, symbol_prices, horizons)

        row = {"event_type": event["event_type"], "symbol": event["symbol"]}
        row.update({f"return_{h}d": forward[h] for h in horizons})
        rows.append(row)

    columns = ["event_type", "symbol"] + [f"return_{h}d" for h in horizons]
    return pd.DataFrame(rows, columns=columns)


def summarize_event_impact(
    occurrence_returns: pd.DataFrame, horizons: tuple[int, ...] = (1, 5, 20)
) -> pd.DataFrame:
    """Aggregates per-occurrence forward returns into a mean-impact table per event type."""
    return_cols = [f"return_{h}d" for h in horizons]
    if occurrence_returns.empty:
        return pd.DataFrame(columns=["event_type", "occurrences", *[f"mean_{c}" for c in return_cols]])

    agg_kwargs = {"occurrences": ("symbol", "count")}
    agg_kwargs.update({f"mean_{c}": (c, "mean") for c in return_cols})
    return occurrence_returns.groupby("event_type").agg(**agg_kwargs).reset_index()
