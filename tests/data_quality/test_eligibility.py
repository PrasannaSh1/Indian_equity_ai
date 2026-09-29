import numpy as np
import pandas as pd

from src.data_quality.eligibility import MIN_HISTORY_TRADING_DAYS, assess_market_data_eligibility
from src.technical.indicators import add_technical_indicators
from src.technical.score import technical_score


def _synthetic_indicator_df(symbol: str, n_days: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    steps = rng.normal(loc=0.0003, scale=0.015, size=n_days)
    close = 100 * np.cumprod(1 + steps)
    ohlcv = pd.DataFrame(
        {
            "date": dates,
            "symbol": symbol,
            "open": close * (1 - 0.002),
            "high": close * (1 + 0.01),
            "low": close * (1 - 0.01),
            "close": close,
            "adjusted_close": close,
            "volume": rng.integers(1_000_000, 5_000_000, size=n_days),
        }
    )
    indicators = add_technical_indicators(ohlcv)
    indicators["technical_score"] = technical_score(indicators)
    return indicators


def test_insufficient_history_is_not_ml_eligible():
    df = _synthetic_indicator_df("X", n_days=50)

    report = assess_market_data_eligibility(df)

    assert report.ml_eligible is False
    assert report.feature_complete is False
    assert "trading days" in report.reason


def test_sufficient_history_is_ml_eligible():
    df = _synthetic_indicator_df("X", n_days=MIN_HISTORY_TRADING_DAYS + 60)

    report = assess_market_data_eligibility(df)

    assert report.ml_eligible is True
    assert report.feature_complete is True
    assert report.missing_features == []
    assert report.reason is None


def test_eligibility_report_to_dict_is_json_shaped():
    df = _synthetic_indicator_df("X", n_days=50)
    report = assess_market_data_eligibility(df)

    payload = report.to_dict()

    assert set(payload) == {
        "history_days",
        "history_years",
        "feature_complete",
        "ml_eligible",
        "missing_features",
        "reason",
    }
