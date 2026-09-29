"""Database schema for the Phase 13 application.

SQLite for now (see PROJECT_PLAN.md Phase 13 notes): standing up a live
PostgreSQL service is a heavier infrastructure step than this phase needs,
and this project's data is small and read-mostly. Using SQLAlchemy Core keeps
the table definitions portable -- switching to Postgres later is a matter of
changing the connection URL, not rewriting the schema, since none of the
column types used here are SQLite-specific.

Tables are a subset of the full schema in Indian_Equity_AI_Project_Plan.md
Section 7, restricted to what this project actually has real, computed data
for (Phases 1-12) -- no empty placeholder tables for data this project never
collected (e.g. prices_intraday, corporate_actions).
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
)

DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "app.db"

metadata = MetaData()

prices_daily = Table(
    "prices_daily",
    metadata,
    Column("date", Date, primary_key=True),
    Column("symbol", String, primary_key=True),
    Column("open", Float),
    Column("high", Float),
    Column("low", Float),
    Column("close", Float),
    Column("adjusted_close", Float),
    Column("volume", Float),
)

technical_features = Table(
    "technical_features",
    metadata,
    Column("date", Date, primary_key=True),
    Column("symbol", String, primary_key=True),
    Column("close", Float),
    Column("sma_20", Float),
    Column("sma_50", Float),
    Column("sma_200", Float),
    Column("rsi_14", Float),
    Column("macd", Float),
    Column("macd_signal", Float),
    Column("macd_histogram", Float),
    Column("atr_14", Float),
    Column("bb_width", Float),
    Column("volume_ratio", Float),
    Column("technical_score", Float),
)

fundamentals_snapshot = Table(
    "fundamentals_snapshot",
    metadata,
    Column("symbol", String, primary_key=True),
    Column("roe", Float),
    Column("roce", Float),
    Column("debt_to_equity", Float),
    Column("current_ratio", Float),
    Column("revenue_growth_yoy", Float),
    Column("pe_ratio", Float),
    Column("pb_ratio", Float),
    Column("dividend_yield", Float),
    Column("fundamental_score", Float),
)

news = Table(
    "news",
    metadata,
    Column("news_id", String, primary_key=True),
    Column("symbol", String, primary_key=True),
    Column("headline", String),
    Column("source", String),
    Column("published_timestamp", String),
    Column("positive_probability", Float),
    Column("negative_probability", Float),
    Column("event_types", String),
)

model_predictions = Table(
    "model_predictions",
    metadata,
    Column("date", Date, primary_key=True),
    Column("symbol", String, primary_key=True),
    Column("close", Float),
    Column("probability_up", Float),
    Column("expected_volatility", Float),
    Column("price_q10", Float),
    Column("price_q50", Float),
    Column("price_q90", Float),
    Column("risk_score", Float),
    Column("risk_label", String),
)

explanations = Table(
    "explanations",
    metadata,
    Column("date", Date, primary_key=True),
    Column("symbol", String, primary_key=True),
    Column("probability_up", Float),
    Column("confidence", Float),
    Column("confidence_label", String),
)

entry_exit = Table(
    "entry_exit",
    metadata,
    Column("date", Date, primary_key=True),
    Column("symbol", String, primary_key=True),
    Column("has_signal", Boolean),
    Column("entry_low", Float),
    Column("entry_high", Float),
    Column("stop", Float),
    Column("target", Float),
)

backtest_results = Table(
    "backtest_results",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("system", String),
    Column("metric", String),
    Column("value", Float),
)


def get_engine(db_path: Path = DEFAULT_DB_PATH):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{db_path}")


def create_all_tables(engine) -> None:
    metadata.create_all(engine)
