"""Horizon-specific targets: swing/short-term and long-term (project plan Section 26).

Intraday is deliberately out of scope: this project only has daily EOD data.
Section 26's intraday feature list (1/5/15-minute bars, intraday VWAP) would
need a different, likely paid, real-time/historical intraday data source --
not something to fabricate. Swing/short-term and long-term are both built on
the daily data already collected, at different forward horizons and with
different feature emphasis (technical for swing, fundamental for long-term).
"""

from __future__ import annotations

import pandas as pd

SWING_HORIZON_DAYS = 5
LONG_TERM_HORIZON_DAYS = 60


def add_horizon_target(df: pd.DataFrame, horizon_days: int, target_col: str) -> pd.DataFrame:
    """Adds {target_col} = 1 if close[t+horizon_days] > close[t] else 0, per symbol.

    Rows without a full forward window get NaN, not a fabricated 0 -- the same
    NaN-comparison pitfall Phase 5's add_target was fixed for (a naive
    `(future_close > close)` cast treats pandas' NaN comparison, which
    evaluates to False, as a real "down" label unless explicitly masked).
    """
    df = df.sort_values(["symbol", "date"]).copy()
    future_close = df.groupby("symbol")["close"].shift(-horizon_days)
    direction = (future_close > df["close"]).astype(float)
    df[target_col] = direction.where(future_close.notna())
    return df
