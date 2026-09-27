"""Drawdown, beta, and correlation calculations for stock analytics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_drawdown(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a 'drawdown' column: close vs the running peak close, per symbol.

    drawdown is <= 0, e.g. -0.15 means 15% below the running peak.
    """
    df = df.sort_values(["symbol", "date"]).copy()
    running_peak = df.groupby("symbol")["close"].cummax()
    df["drawdown"] = df["close"] / running_peak - 1
    return df


def max_drawdown(df: pd.DataFrame) -> pd.Series:
    """Returns the maximum (most negative) drawdown per symbol."""
    if "drawdown" not in df.columns:
        df = add_drawdown(df)
    return df.groupby("symbol")["drawdown"].min()


def period_returns(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Resamples close prices to a calendar frequency and computes the period return.

    freq examples: 'W-FRI' for weekly (Friday close-to-close), 'ME' for monthly.
    Returns a long DataFrame with columns [date, symbol, return].
    """
    out = []
    for symbol, group in df.groupby("symbol"):
        s = group.set_index("date")["close"].resample(freq).last().dropna()
        out.append(pd.DataFrame({"date": s.index, "symbol": symbol, "return": s.pct_change().values}))
    return pd.concat(out, ignore_index=True)


def beta(stock_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Computes beta = Cov(stock, benchmark) / Var(benchmark) on the aligned, non-NaN overlap."""
    aligned = pd.concat(
        [stock_returns.rename("stock"), benchmark_returns.rename("benchmark")],
        axis=1,
        join="inner",
    ).dropna()
    if len(aligned) < 2:
        return float("nan")
    covariance = aligned["stock"].cov(aligned["benchmark"])
    variance = aligned["benchmark"].var()
    if variance == 0 or np.isnan(variance):
        return float("nan")
    return covariance / variance


def correlation_matrix(returns_wide: pd.DataFrame) -> pd.DataFrame:
    """Pairwise correlation of daily returns, columns = symbols/benchmarks."""
    return returns_wide.corr()
