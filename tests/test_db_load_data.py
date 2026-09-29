import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from src.db.load_data import load_all, load_prices_daily
from src.db.schema import create_all_tables


@pytest.fixture
def memory_engine():
    # StaticPool ensures every connection checked out from this engine shares the
    # same underlying in-memory database (plain sqlite:///:memory: gives each new
    # connection its own empty database, which is fragile if anything here ever
    # ends up on a different thread/connection than the one that created the tables).
    engine = create_engine(
        "sqlite:///:memory:", poolclass=StaticPool, connect_args={"check_same_thread": False}
    )
    create_all_tables(engine)
    return engine


def test_load_prices_daily_round_trips_correctly(tmp_path, memory_engine):
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "symbol": ["A", "A"],
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "adjusted_close": [101.0, 102.0],
            "volume": [1000.0, 1100.0],
        }
    )
    df.to_parquet(tmp_path / "prices_daily.parquet", index=False)

    count = load_prices_daily(memory_engine, tmp_path)

    assert count == 2
    stored = pd.read_sql("SELECT * FROM prices_daily ORDER BY date", memory_engine)
    assert stored["close"].tolist() == pytest.approx([101.0, 102.0])
    assert stored["symbol"].tolist() == ["A", "A"]


def _write_minimal_fixtures(tmp_path):
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"]), "symbol": ["A"], "open": [100.0], "high": [101.0],
            "low": [99.0], "close": [100.5], "adjusted_close": [100.5], "volume": [1000.0],
        }
    )
    prices.to_parquet(tmp_path / "prices_daily.parquet", index=False)

    technical_cols = [
        "date", "symbol", "open", "high", "low", "close", "volume",
        "sma_20", "sma_50", "sma_200", "ema_20", "ema_50", "rsi_14",
        "macd", "macd_signal", "macd_histogram", "atr_14",
        "bb_upper", "bb_middle", "bb_lower", "bb_width", "volume_ratio", "technical_score",
    ]
    technical = pd.DataFrame([[pd.Timestamp("2024-01-01"), "A"] + [1.0] * (len(technical_cols) - 2)], columns=technical_cols)
    technical.to_parquet(tmp_path / "technical_features.parquet", index=False)

    fundamentals = pd.DataFrame(
        {
            "roe": [0.15], "roce": [0.2], "debt_to_equity": [0.5], "current_ratio": [1.5],
            "revenue_growth_yoy": [0.1], "pe_ratio": [20.0], "pb_ratio": [3.0],
            "dividend_yield": [0.01], "fundamental_score": [60.0],
        },
        index=["A"],
    )
    fundamentals.to_csv(tmp_path / "fundamentals_snapshot.csv")

    news = pd.DataFrame(
        {
            "news_id": ["n1"], "symbol": ["A"], "headline": ["headline"], "source": ["Reuters"],
            "published_timestamp": ["2024-01-01T00:00:00Z"], "positive_probability": [0.5],
            "negative_probability": [0.2], "event_types": ["earnings"],
        }
    )
    news.to_csv(tmp_path / "news_scored.csv", index=False)

    predictions = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-01")], "symbol": ["A"], "close": [100.5],
            "probability_up": [0.6], "expected_volatility": [0.02], "price_q10": [98.0],
            "price_q50": [100.5], "price_q90": [103.0], "risk_score": [40.0], "risk_label": ["Medium"],
        }
    )
    predictions.to_parquet(tmp_path / "risk_engine_output.parquet", index=False)

    explanations = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-01")], "symbol": ["A"], "close": [100.5],
            "probability_up": [0.6], "confidence": [0.2], "confidence_label": ["Low"],
        }
    )
    explanations.to_parquet(tmp_path / "explainability_predictions.parquet", index=False)

    entry_exit = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-01")], "symbol": ["A"], "close": [100.5],
            "probability_up": [0.6], "has_signal": [True], "entry_low": [100.5],
            "entry_high": [101.0], "stop": [98.0], "target": [104.0],
        }
    )
    entry_exit.to_parquet(tmp_path / "entry_exit_output.parquet", index=False)

    backtest = pd.DataFrame(
        {"strategy_net_of_cost": [0.1, -0.2], "nifty50_buy_and_hold": [0.05, -0.05]},
        index=["cumulative_return", "sharpe_ratio"],
    )
    backtest.to_csv(tmp_path / "backtest_comparison.csv")


def test_load_all_loads_every_table_with_expected_row_counts(tmp_path, memory_engine):
    _write_minimal_fixtures(tmp_path)

    counts = load_all(data_processed=tmp_path, engine=memory_engine)

    assert counts["prices_daily"] == 1
    assert counts["technical_features"] == 1
    assert counts["fundamentals_snapshot"] == 1
    assert counts["news"] == 1
    assert counts["model_predictions"] == 1
    assert counts["explanations"] == 1
    assert counts["entry_exit"] == 1
    assert counts["backtest_results"] == 4  # 2 metrics x 2 systems, melted long

    stored_fundamentals = pd.read_sql("SELECT * FROM fundamentals_snapshot", memory_engine)
    assert stored_fundamentals["symbol"].tolist() == ["A"]
    assert stored_fundamentals["fundamental_score"].iloc[0] == pytest.approx(60.0)
