import pandas as pd
import pytest

from src.backtesting.engine import (
    aggregate_portfolio_returns,
    compute_itemized_transaction_costs,
    generate_signals,
    simulate_trades,
)


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


def test_compute_itemized_transaction_costs_breaks_down_to_the_documented_total():
    breakdown = compute_itemized_transaction_costs()

    # brokerage(0) + stt(10) + exchange(0.35) + sebi(0.01) + stamp_duty(1.5) +
    # gst(0.18*(0+0.36)=0.0648) + slippage(5) = 16.9248
    assert breakdown["total_bps"] == pytest.approx(16.9248)
    assert breakdown["gst_bps"] == pytest.approx(0.18 * (0.35 + 0.01))
    component_sum = (
        breakdown["brokerage_bps"]
        + breakdown["stt_bps"]
        + breakdown["exchange_txn_bps"]
        + breakdown["sebi_turnover_bps"]
        + breakdown["stamp_duty_bps"]
        + breakdown["gst_bps"]
        + breakdown["slippage_bps"]
    )
    assert component_sum == pytest.approx(breakdown["total_bps"])


def test_simulate_trades_backward_compatible_when_bps_passed_explicitly():
    # Pinning the exact pre-itemization hand-computed case: explicit bps must
    # still produce the old flat combined-cost result, unaffected by itemization.
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "symbol": ["A", "B"],
            "probability_up": [0.6, 0.4],
            "next_day_return": [0.02, 0.05],
        }
    )
    out = simulate_trades(df, threshold=0.5, transaction_cost_bps=10, slippage_bps=5)

    assert out["cost"].tolist() == pytest.approx([0.0015, 0.0])
    assert out["net_return"].tolist() == pytest.approx([0.0185, 0.0])


def test_simulate_trades_defaults_to_itemized_cost_when_bps_omitted():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"]),
            "symbol": ["A"],
            "probability_up": [0.6],
            "next_day_return": [0.02],
        }
    )
    out = simulate_trades(df, threshold=0.5)

    expected_cost = compute_itemized_transaction_costs()["total_bps"] / 10_000
    assert out["cost"].iloc[0] == pytest.approx(expected_cost)
    assert out["net_return"].iloc[0] == pytest.approx(0.02 - expected_cost)


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
