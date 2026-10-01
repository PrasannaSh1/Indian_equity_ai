import pandas as pd
import pytest

from src.backtesting.metrics import (
    active_position_rate,
    annualized_return,
    calmar_ratio,
    cumulative_return,
    max_drawdown_from_returns,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    turnover,
    underperforms_benchmark,
    win_rate,
)


def test_cumulative_return_matches_hand_computed_compounding():
    returns = pd.Series([0.1, -0.1, 0.1])
    # 1.1 * 0.9 * 1.1 - 1 = 1.089 - 1
    assert cumulative_return(returns) == pytest.approx(0.089)


def test_annualized_return_matches_hand_computed_geometric_scaling():
    # total_growth = 1.1*1.1 = 1.21; exponent = periods_per_year/n = 1/2 -> sqrt(1.21) = 1.1
    returns = pd.Series([0.1, 0.1])
    assert annualized_return(returns, periods_per_year=1) == pytest.approx(0.1)


def test_sharpe_ratio_matches_hand_computed_mean_over_std():
    # mean=0.10, sample std (ddof=1) = 0.02 exactly
    returns = pd.Series([0.08, 0.10, 0.12])
    assert sharpe_ratio(returns, periods_per_year=1) == pytest.approx(5.0)


def test_sortino_ratio_only_penalizes_downside_deviation():
    # mean(excess) over all 4 = 0.03; downside = [-0.02,-0.04,-0.06], sample std = 0.02
    returns = pd.Series([-0.02, -0.04, -0.06, 0.24])
    assert sortino_ratio(returns, periods_per_year=1) == pytest.approx(1.5)


def test_max_drawdown_from_returns_matches_hand_computed_peak_to_trough():
    returns = pd.Series([0.20, -0.50, 0.0])
    # equity: 1.2, 0.6, 0.6 ; peak: 1.2, 1.2, 1.2 -> drawdown min = -0.5
    assert max_drawdown_from_returns(returns) == pytest.approx(-0.5)


def test_calmar_ratio_matches_hand_computed_ratio():
    # n == periods_per_year -> annualized_return == cumulative_return == -0.4; max_drawdown == -0.5
    returns = pd.Series([0.20, -0.50])
    assert calmar_ratio(returns, periods_per_year=2) == pytest.approx(-0.8)


def test_win_rate_and_profit_factor_match_hand_computed_values():
    trade_returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.0])
    assert win_rate(trade_returns) == pytest.approx(0.4)  # 2 of 5 are strictly positive
    assert profit_factor(trade_returns) == pytest.approx(4 / 3)  # 0.04 gains / 0.03 losses


def test_turnover_is_the_mean_of_the_signal_column():
    signal = pd.Series([1, 0, 1, 1, 0])
    assert turnover(signal) == pytest.approx(0.6)


def test_active_position_rate_matches_deprecated_turnover_alias():
    signal = pd.Series([1, 0, 1, 1, 0])
    assert active_position_rate(signal) == turnover(signal) == pytest.approx(0.6)


def test_underperforms_benchmark_flags_strategy_worse_than_index():
    # The project's actual documented backtest result: strategy loses, benchmark doesn't.
    assert underperforms_benchmark(strategy_metric=-0.649, benchmark_metric=-0.078) is True


def test_underperforms_benchmark_false_when_strategy_beats_benchmark():
    assert underperforms_benchmark(strategy_metric=0.10, benchmark_metric=0.05) is False


def test_underperforms_benchmark_respects_higher_is_better_false():
    # e.g. comparing a cost ratio, where a LOWER number is the better outcome.
    assert underperforms_benchmark(strategy_metric=0.02, benchmark_metric=0.01, higher_is_better=False) is True
    assert underperforms_benchmark(strategy_metric=0.01, benchmark_metric=0.02, higher_is_better=False) is False
