import numpy as np
import pandas as pd

from src.models.dataset import FEATURE_SCHEMA_VERSION
from src.models.registry import load_all
from src.models.train_global import train_global_model
from src.technical.indicators import add_technical_indicators
from src.technical.score import technical_score


def _synthetic_technical_features(symbols, n_days=320, seed=0) -> pd.DataFrame:
    """A small multi-symbol technical_features.parquet stand-in: enough trading
    days per symbol for the slowest indicator (SMA-200) to warm up, but far smaller
    than the real 50-stock/5-year dataset, so training here stays fast.
    """
    rng = np.random.default_rng(seed)
    frames = []
    for i, symbol in enumerate(symbols):
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


def test_train_global_model_persists_a_model_and_registers_it(tmp_path):
    technical = _synthetic_technical_features(["AAA", "BBB", "CCC"])
    data_processed = tmp_path / "processed"
    data_processed.mkdir()
    technical.to_parquet(data_processed / "technical_features.parquet")
    # Mirrors the real REPO_ROOT/data/models layout (two levels under the root) so
    # train_global.py's model_path-relative-to-repo-root logic resolves the same way.
    model_dir = tmp_path / "data" / "models"

    entry = train_global_model(data_processed=data_processed, model_dir=model_dir)

    assert entry["feature_schema_version"] == FEATURE_SCHEMA_VERSION
    assert entry["model_name"] in {
        "naive_baseline",
        "logistic_regression",
        "random_forest",
        "xgboost",
        "lightgbm",
    }
    assert "roc_auc" in entry["test_metrics"]
    assert (model_dir / "registry.json").exists()

    if entry["model_name"] != "naive_baseline":
        assert entry["model_path"] is not None
        assert (tmp_path / entry["model_path"]).exists()
    else:
        assert entry["model_path"] is None

    registered = load_all(model_dir=model_dir)
    assert any(e["model_version"] == entry["model_version"] for e in registered)
