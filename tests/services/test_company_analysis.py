from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.data_quality.eligibility import MIN_HISTORY_TRADING_DAYS
from src.models.dataset import FEATURE_SCHEMA_VERSION
import src.services.company_analysis as company_analysis


def _synthetic_ohlcv(symbol: str, n_days: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    steps = rng.normal(loc=0.0003, scale=0.015, size=n_days)
    close = 100 * np.cumprod(1 + steps)
    return pd.DataFrame(
        {
            "date": dates,
            "symbol": symbol,
            "open": close * 0.998,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "adjusted_close": close,
            "volume": rng.integers(1_000_000, 5_000_000, size=n_days),
        }
    )


def _fake_entry_model_scaler():
    entry = {
        "model_version": "test_v1",
        "model_name": "logistic_regression",
        "needs_scaling": False,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "base_rate": 0.5,
    }
    model = SimpleNamespace(predict_proba=lambda X: np.tile([0.4, 0.6], (len(X), 1)))
    return entry, model, None


@pytest.fixture(autouse=True)
def _stub_external_calls(monkeypatch):
    """Every test in this module runs offline: no real yfinance/model-registry calls."""

    def fake_download(symbol, period="5y"):
        return _synthetic_ohlcv(symbol, n_days=MIN_HISTORY_TRADING_DAYS + 60)

    def fake_index_download(ticker, period="5y"):
        return _synthetic_ohlcv(ticker, n_days=MIN_HISTORY_TRADING_DAYS + 60, seed=1)

    monkeypatch.setattr(company_analysis, "download_daily_ohlcv", fake_download)
    monkeypatch.setattr(company_analysis, "download_index_ohlcv", fake_index_download)
    monkeypatch.setattr(company_analysis, "load_model", lambda model_version=None: _fake_entry_model_scaler())


def test_analyze_known_company_returns_full_prediction_pipeline():
    result = company_analysis.analyze_company("TCS", horizon="5d", analysis_type="quick")

    assert result["company"]["symbol"] == "TCS"
    assert result["model_coverage"]["in_training_universe"] is True
    assert result["data_quality"]["ml_eligible"] is True
    assert result["prediction"]["available"] is True
    assert result["prediction"]["probability_up"] == pytest.approx(0.6)
    assert result["entry_exit"]["available"] is True
    assert result["risk"]["available"] is True


def test_analyze_unseen_company_flags_not_in_training_universe():
    result = company_analysis.analyze_company("DIXON", horizon="5d", analysis_type="quick")

    assert result["company"]["symbol"] == "DIXON"
    assert result["model_coverage"]["in_training_universe"] is False
    assert "NOT retrained" in result["model_coverage"]["note"]
    assert result["prediction"]["available"] is True


def test_analyze_company_with_insufficient_history_skips_prediction(monkeypatch):
    monkeypatch.setattr(
        company_analysis,
        "download_daily_ohlcv",
        lambda symbol, period="5y": _synthetic_ohlcv(symbol, n_days=50),
    )

    result = company_analysis.analyze_company("TCS", horizon="5d", analysis_type="quick")

    assert result["data_quality"]["ml_eligible"] is False
    assert result["prediction"]["available"] is False
    assert "INSUFFICIENT_DATA" in result["prediction"]["reason"]
    assert result["entry_exit"]["available"] is False


def test_analyze_company_quick_skips_fundamentals_news_rag():
    result = company_analysis.analyze_company("TCS", analysis_type="quick")

    assert result["fundamentals"]["available"] is False
    assert result["sentiment"]["available"] is False
    assert result["rag"]["available"] is False


def test_analyze_company_rejects_unsupported_horizon():
    with pytest.raises(ValueError):
        company_analysis.analyze_company("TCS", horizon="15d")
