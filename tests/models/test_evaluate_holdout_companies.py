import numpy as np
import pandas as pd

from src.models.evaluate_holdout_companies import evaluate_holdout_generalization, split_universe
from src.technical.indicators import add_technical_indicators
from src.technical.score import technical_score


def _synthetic_technical_features(symbols, n_days=320, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for symbol in symbols:
        dates = pd.bdate_range("2020-01-01", periods=n_days)
        steps = rng.normal(loc=0.0002, scale=0.015, size=n_days)
        close = 100 * np.cumprod(1 + steps)
        ohlcv = pd.DataFrame(
            {
                "date": dates,
                "symbol": symbol,
                "open": close * 0.998,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": rng.integers(1_000_000, 5_000_000, size=n_days),
            }
        )
        indicators = add_technical_indicators(ohlcv)
        indicators["technical_score"] = technical_score(indicators)
        frames.append(indicators)
    return pd.concat(frames, ignore_index=True)


def test_split_universe_partitions_without_overlap():
    universe = [f"SYM{i}" for i in range(10)]

    train, holdout = split_universe(universe, holdout_fraction=0.2, seed=42)

    assert set(train) & set(holdout) == set()
    assert set(train) | set(holdout) == set(universe)
    assert len(holdout) == 2


def test_evaluate_holdout_generalization_produces_seen_and_unseen_metrics(tmp_path):
    symbols = [f"SYM{i}" for i in range(6)]
    technical = _synthetic_technical_features(symbols)
    data_processed = tmp_path / "processed"
    data_processed.mkdir()
    technical.to_parquet(data_processed / "technical_features.parquet")

    result = evaluate_holdout_generalization(
        data_processed=data_processed,
        holdout_fraction=0.34,
        universe=symbols,
    )

    assert set(result["train_tickers"]) & set(result["holdout_tickers"]) == set()
    assert "roc_auc" in result["seen_companies_unseen_dates_metrics"]
    assert "roc_auc" in result["unseen_companies_metrics"]
    assert (data_processed / "holdout_company_generalization_metrics.csv").exists()
