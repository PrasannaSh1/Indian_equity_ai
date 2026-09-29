"""Signal-driven backtest simulation with transaction costs and slippage.

Trading convention (deliberately simple, documented rather than hidden): a
signal generated from day t's features is "entered" at day t's close and
"exited" at day t+1's close -- i.e. the trade's realized return is exactly
next_day_return, the same quantity the model predicts. This is a
simplification (a real order can't fill at the exact close price used to
generate the signal); it is chosen because it maps directly onto what the
Phase 5/7/8 models forecast, and the cost/slippage assumptions below are
intended to compensate for some of that optimism.

Cost assumptions (round-trip, i.e. covering both the entry and the exit leg):
- transaction_cost_bps=10 (0.10%): brokerage + STT + exchange charges + GST +
  stamp duty for a typical Indian retail delivery trade, combined.
- slippage_bps=5 (0.05%): a conservative estimate of price impact/spread cost
  for liquid large-cap NSE stocks.
Both are configurable; the defaults are a reasonable, stated assumption, not a
precise brokerage quote.
"""

from __future__ import annotations

import pandas as pd

DEFAULT_TRANSACTION_COST_BPS = 10
DEFAULT_SLIPPAGE_BPS = 5


def generate_signals(probability_up: pd.Series, threshold: float = 0.55) -> pd.Series:
    """Long-only signal: 1 (take the trade) if probability_up > threshold, else 0 (stay in cash)."""
    return (probability_up > threshold).astype(int)


def simulate_trades(
    df: pd.DataFrame,
    threshold: float = 0.55,
    transaction_cost_bps: float = DEFAULT_TRANSACTION_COST_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
) -> pd.DataFrame:
    """Requires columns: date, symbol, next_day_return, probability_up.

    Adds: signal, gross_return (next_day_return if signal==1, else 0),
    cost (round-trip friction, only charged when a trade is actually taken),
    net_return (gross_return - cost).
    """
    df = df.copy()
    df["signal"] = generate_signals(df["probability_up"], threshold)
    df["gross_return"] = df["next_day_return"] * df["signal"]

    round_trip_cost = (transaction_cost_bps + slippage_bps) / 10_000
    df["cost"] = round_trip_cost * df["signal"]
    df["net_return"] = df["gross_return"] - df["cost"]
    return df


def aggregate_portfolio_returns(trades: pd.DataFrame) -> pd.Series:
    """Equal-weights net_return across every symbol with an active signal that day;
    a day with no active signals anywhere returns 0 (fully in cash).
    """
    active = trades[trades["signal"] == 1]
    daily_return = active.groupby("date")["net_return"].mean()

    all_dates = pd.Series(trades["date"].unique()).sort_values()
    return daily_return.reindex(all_dates, fill_value=0.0).sort_index()
