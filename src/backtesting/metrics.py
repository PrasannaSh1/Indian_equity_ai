"""Trading performance metrics for a daily return series."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def cumulative_return(returns: pd.Series) -> float:
    return float((1 + returns).prod() - 1)


def annualized_return(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    n = len(returns)
    if n == 0:
        return float("nan")
    total_growth = (1 + returns).prod()
    return float(total_growth ** (periods_per_year / n) - 1)


def sharpe_ratio(
    returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR, risk_free_rate: float = 0.0
) -> float:
    daily_rf = risk_free_rate / periods_per_year
    excess = returns - daily_rf
    std = excess.std()
    if std == 0 or np.isnan(std):
        return float("nan")
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR, risk_free_rate: float = 0.0
) -> float:
    daily_rf = risk_free_rate / periods_per_year
    excess = returns - daily_rf
    downside = excess[excess < 0]
    downside_std = downside.std()
    if downside_std == 0 or np.isnan(downside_std):
        return float("nan")
    return float(excess.mean() / downside_std * np.sqrt(periods_per_year))


def max_drawdown_from_returns(returns: pd.Series) -> float:
    equity_curve = (1 + returns).cumprod()
    running_peak = equity_curve.cummax()
    drawdown = equity_curve / running_peak - 1
    return float(drawdown.min())


def calmar_ratio(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    mdd = max_drawdown_from_returns(returns)
    if mdd == 0:
        return float("nan")
    return float(annualized_return(returns, periods_per_year) / abs(mdd))


def win_rate(trade_returns: pd.Series) -> float:
    """`trade_returns` should already be filtered to only rows where a trade was taken."""
    if len(trade_returns) == 0:
        return float("nan")
    return float((trade_returns > 0).mean())


def profit_factor(trade_returns: pd.Series) -> float:
    gains = trade_returns[trade_returns > 0].sum()
    losses = trade_returns[trade_returns < 0].sum()
    if losses == 0:
        return float("nan")
    return float(gains / abs(losses))


def active_position_rate(signal: pd.Series) -> float:
    """Fraction of (symbol, day) observations with an active position."""
    return float(signal.mean())


def turnover(signal: pd.Series) -> float:
    """Deprecated alias for active_position_rate -- kept for backward
    compatibility. "Turnover" is a misleading name for this quantity (website
    audit Section 25): it measures position *occupancy* (what fraction of
    rows hold a position), not the traditional portfolio-turnover sense of how
    much of the portfolio is bought/sold per period, which this project's
    backtest doesn't separately track (it has no absolute position-size state
    between days, only a daily hold/no-hold signal per symbol).
    """
    return active_position_rate(signal)


def underperforms_benchmark(strategy_metric: float, benchmark_metric: float, higher_is_better: bool = True) -> bool:
    """True if the strategy's metric is worse than the benchmark's (website audit
    Section 24: this must drive a prominent warning, not a result buried in a tab).
    """
    if higher_is_better:
        return strategy_metric < benchmark_metric
    return strategy_metric > benchmark_metric
