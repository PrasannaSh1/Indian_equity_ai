"""ETL: loads Phases 1-12's computed outputs (data/processed/*) into the app database."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.db.schema import create_all_tables, get_engine

DEFAULT_DATA_PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"


def load_prices_daily(engine, data_processed: Path) -> int:
    df = pd.read_parquet(data_processed / "prices_daily.parquet")
    df = df[["date", "symbol", "open", "high", "low", "close", "adjusted_close", "volume"]]
    df.to_sql("prices_daily", engine, if_exists="replace", index=False)
    return len(df)


def load_technical_features(engine, data_processed: Path) -> int:
    df = pd.read_parquet(data_processed / "technical_features.parquet")
    cols = [
        "date", "symbol", "open", "high", "low", "close", "volume",
        "sma_20", "sma_50", "sma_200", "ema_20", "ema_50", "rsi_14",
        "macd", "macd_signal", "macd_histogram", "atr_14",
        "bb_upper", "bb_middle", "bb_lower", "bb_width",
        "volume_ratio", "technical_score",
    ]
    df = df[cols]
    df.to_sql("technical_features", engine, if_exists="replace", index=False)
    return len(df)


def load_fundamentals_snapshot(engine, data_processed: Path) -> int:
    df = pd.read_csv(data_processed / "fundamentals_snapshot.csv", index_col=0)
    df = df.reset_index().rename(columns={"index": "symbol"})
    cols = [
        "symbol", "roe", "roce", "debt_to_equity", "current_ratio",
        "revenue_growth_yoy", "pe_ratio", "pb_ratio", "dividend_yield", "fundamental_score",
    ]
    df = df[cols]
    df.to_sql("fundamentals_snapshot", engine, if_exists="replace", index=False)
    return len(df)


def load_news(engine, data_processed: Path) -> int:
    df = pd.read_csv(data_processed / "news_scored.csv")
    cols = [
        "news_id", "symbol", "headline", "source", "published_timestamp",
        "positive_probability", "negative_probability", "event_types",
    ]
    df = df[cols]
    df.to_sql("news", engine, if_exists="replace", index=False)
    return len(df)


def load_model_predictions(engine, data_processed: Path) -> int:
    df = pd.read_parquet(data_processed / "risk_engine_output.parquet")
    df.to_sql("model_predictions", engine, if_exists="replace", index=False)
    return len(df)


def load_explanations(engine, data_processed: Path) -> int:
    df = pd.read_parquet(data_processed / "explainability_predictions.parquet")
    df = df[["date", "symbol", "probability_up", "confidence", "confidence_label"]]
    df.to_sql("explanations", engine, if_exists="replace", index=False)
    return len(df)


def load_entry_exit(engine, data_processed: Path) -> int:
    df = pd.read_parquet(data_processed / "entry_exit_output.parquet")
    df = df[["date", "symbol", "has_signal", "entry_low", "entry_high", "stop", "target"]]
    df.to_sql("entry_exit", engine, if_exists="replace", index=False)
    return len(df)


def load_backtest_results(engine, data_processed: Path) -> int:
    df = pd.read_csv(data_processed / "backtest_comparison.csv", index_col=0)
    long = (
        df.reset_index()
        .melt(id_vars="index", var_name="system", value_name="value")
        .rename(columns={"index": "metric"})
    )
    long = long[["system", "metric", "value"]]
    long.to_sql("backtest_results", engine, if_exists="replace", index=False)
    return len(long)


LOADERS = {
    "prices_daily": load_prices_daily,
    "technical_features": load_technical_features,
    "fundamentals_snapshot": load_fundamentals_snapshot,
    "news": load_news,
    "model_predictions": load_model_predictions,
    "explanations": load_explanations,
    "entry_exit": load_entry_exit,
    "backtest_results": load_backtest_results,
}


def load_all(data_processed: Path = DEFAULT_DATA_PROCESSED, engine=None) -> dict[str, int]:
    engine = engine if engine is not None else get_engine()
    create_all_tables(engine)
    return {name: loader(engine, data_processed) for name, loader in LOADERS.items()}


if __name__ == "__main__":
    counts = load_all()
    for table, count in counts.items():
        print(f"{table}: {count} rows")
