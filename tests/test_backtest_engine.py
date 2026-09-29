import pandas as pd
import pytest

from src.backtesting.engine import aggregate_portfolio_returns, generate_signals, simulate_trades


def test_generate_signals_thresholds_correctly():
    proba = pd.Series([0.6, 0.5, 0.4, 0.56])
    out = generate_signals(proba, threshold=0.55)
    assert out.tolist() == [1, 0, 0, 1]


def test_simulate_trades_matches_hand_computed_costs():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "symbol": ["A", "B"],
            "probability_up": [0.6, 0.4],
            "next_day_return": [0.02, 0.05],  # B's return is irrelevant: no trade taken
        }
    )
    out = simulate_trades(df, threshold=0.5, transaction_cost_bps=10, slippage_bps=5)

    assert out["signal"].tolist() == [1, 0]
    assert out["gross_return"].tolist() == pytest.approx([0.02, 0.0])
    # round-trip cost = (10+5)/10000 = 0.0015, only charged when signal==1
    assert out["cost"].tolist() == pytest.approx([0.0015, 0.0])
    assert out["net_return"].tolist() == pytest.approx([0.0185, 0.0])


def test_aggregate_portfolio_returns_averages_only_active_signals():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]),
            "symbol": ["A", "B", "A", "B"],
            "probability_up": [0.9, 0.1, 0.1, 0.1],  # day1: only A active; day2: nothing active
            "next_day_return": [0.02, 0.10, 0.05, -0.05],
        }
    )
    trades = simulate_trades(df, threshold=0.5, transaction_cost_bps=0, slippage_bps=0)
    portfolio = aggregate_portfolio_returns(trades)

    assert portfolio.loc[pd.Timestamp("2024-01-01")] == pytest.approx(0.02)  # only A's return, B ignored
    assert portfolio.loc[pd.Timestamp("2024-01-02")] == pytest.approx(0.0)  # no active signals -> cash
