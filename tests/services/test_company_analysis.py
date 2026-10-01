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
        "horizon_days": 1,
        "training_data_cutoff_date": "2026-08-31",
    }
    model = SimpleNamespace(predict_proba=lambda X: np.tile([0.4, 0.6], (len(X), 1)))
    return entry, model, None


class _FakeQuantileModel:
    def __init__(self, return_value):
        self._return_value = return_value

    def predict(self, X):
        return np.full(len(X), self._return_value)


def _fake_quantile_entry_and_models():
    entry = {"model_version": "quantile_test_v1", "training_data_cutoff_date": "2026-08-31"}
    models = {0.1: _FakeQuantileModel(-0.02), 0.5: _FakeQuantileModel(0.0), 0.9: _FakeQuantileModel(0.02)}
    return entry, models


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
    monkeypatch.setattr(company_analysis, "load_quantile_models", lambda: _fake_quantile_entry_and_models())


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


def test_analyze_company_includes_price_quantile_forecast_from_same_snapshot():
    result = company_analysis.analyze_company("TCS", horizon="5d", analysis_type="quick")

    prediction = result["prediction"]
    assert prediction["price_quantiles_available"] is True
    assert prediction["quantile_model_version"] == "quantile_test_v1"
    assert prediction["price_q10"] < prediction["price_q50"] < prediction["price_q90"]


def test_analyze_company_discloses_served_horizon_instead_of_fabricating_multiday_forecast():
    result = company_analysis.analyze_company("TCS", horizon="60d", analysis_type="quick")

    prediction = result["prediction"]
    assert prediction["served_horizon_days"] == 1
    assert "1-trading-day-ahead" in prediction["horizon_note"]


def test_analyze_company_prediction_includes_confidence_gated_signal_label():
    # The stubbed model returns probability_up=0.6 -> confidence = |0.6-0.5|*2, which
    # float64 arithmetic puts at ~0.19999999999999996 -- just under CONFIDENCE_LOW_MAX
    # (0.2) -- so classify_signal correctly reports Low Confidence, not Bullish, for
    # what is in fact a near-coin-flip probability. This is the exact scenario the
    # website audit flagged: a bare >0.5 check would have called this "Bullish".
    result = company_analysis.analyze_company("TCS", horizon="5d", analysis_type="quick")

    assert result["prediction"]["signal_label"] == "Low Confidence"


def test_analyze_company_entry_exit_reports_validity():
    result = company_analysis.analyze_company("TCS", horizon="5d", analysis_type="quick")

    assert "valid" in result["entry_exit"]
    assert "invalid_reason" in result["entry_exit"]


def test_analyze_company_builds_context_and_lineage():
    result = company_analysis.analyze_company("DIXON", horizon="5d", analysis_type="quick")

    context = result["context"]
    assert context["symbol"] == "DIXON"
    assert context["in_training_universe"] is False
    assert context["classifier_model_version"] == "test_v1"
    assert context["classifier_training_data_cutoff"] == "2026-08-31"
    assert context["quantile_model_version"] == "quantile_test_v1"

    lineage = result["lineage"]
    categories = {row["category"] for row in lineage}
    assert "market_data" in categories
    assert "model_classifier" in categories
