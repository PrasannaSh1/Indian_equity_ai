import numpy as np
import pandas as pd

from src.models.dataset import FEATURE_SCHEMA_VERSION
from src.models.registry import QUANTILE_REGISTRY_FILENAME, load_all
from src.models.train_quantiles import train_quantile_models
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


def test_train_quantile_models_persists_and_registers_with_coverage_metric(tmp_path):
    technical = _synthetic_technical_features(["AAA", "BBB", "CCC"])
    data_processed = tmp_path / "processed"
    data_processed.mkdir()
    technical.to_parquet(data_processed / "technical_features.parquet")
    model_dir = tmp_path / "data" / "models"

    entry = train_quantile_models(data_processed=data_processed, model_dir=model_dir)

    assert entry["feature_schema_version"] == FEATURE_SCHEMA_VERSION
    assert entry["quantiles"] == [0.1, 0.5, 0.9]
    assert set(entry["model_paths"]) == {"q10", "q50", "q90"}
    for relative_path in entry["model_paths"].values():
        assert (tmp_path / relative_path).exists()

    # Coverage must be reported honestly -- a plausible real value in [0, 1], not
    # hidden or silently clamped to look better than it is.
    assert 0.0 <= entry["test_coverage"] <= 1.0
    assert entry["test_interval_pct"] == 80

    registered = load_all(model_dir=model_dir, filename=QUANTILE_REGISTRY_FILENAME)
    assert any(e["model_version"] == entry["model_version"] for e in registered)
    # Lands in its own registry file, not mixed into the classifier's registry.json.
    assert load_all(model_dir=model_dir) == []
