import numpy as np
import pandas as pd
import pytest

from src.models.multimodal_dataset import (
    FUNDAMENTAL_FEATURE_COLUMNS,
    merge_fundamentals_pit,
    merge_macro_and_regime,
    merge_news_features,
)


def _minimal_fundamentals(symbol, fiscal_year_end, **overrides):
    row = {col: 0.0 for col in FUNDAMENTAL_FEATURE_COLUMNS}
    row.update(overrides)
    row["symbol"] = symbol
    row["fiscal_year_end"] = pd.Timestamp(fiscal_year_end)
    return row


def test_merge_fundamentals_pit_hides_data_until_after_the_reporting_lag():
    # Fiscal year ends 2024-03-31; with a 60-day lag, results become visible 2024-05-30.
    fundamentals_annual = pd.DataFrame(
        [_minimal_fundamentals("A", "2024-03-31", roe=0.20)]
    )
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-05-29", "2024-05-30", "2024-06-01"]),
            "symbol": ["A", "A", "A"],
        }
    )
    out = merge_fundamentals_pit(df, fundamentals_annual)

    assert pd.isna(out["roe"].iloc[0])  # one day before the reporting lag elapses
    assert out["roe"].iloc[1] == pytest.approx(0.20)  # exactly on the availability date
    assert out["roe"].iloc[2] == pytest.approx(0.20)  # after


def test_merge_fundamentals_pit_uses_the_latest_available_fiscal_year_only():
    fundamentals_annual = pd.DataFrame(
        [
            _minimal_fundamentals("A", "2023-03-31", roe=0.10),
            _minimal_fundamentals("A", "2024-03-31", roe=0.20),
        ]
    )
    df = pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-12-01"]), "symbol": ["A", "A"]})
    out = merge_fundamentals_pit(df, fundamentals_annual)

    # 2024-01-01 is after FY23's lag (2023-05-30) but before FY24's lag (2024-05-30)
    assert out["roe"].iloc[0] == pytest.approx(0.10)
    assert out["roe"].iloc[1] == pytest.approx(0.20)


def test_merge_fundamentals_pit_does_not_leak_across_symbols():
    fundamentals_annual = pd.DataFrame(
        [
            _minimal_fundamentals("A", "2024-03-31", roe=0.20),
            _minimal_fundamentals("B", "2024-03-31", roe=0.99),
        ]
    )
    df = pd.DataFrame({"date": pd.to_datetime(["2024-06-01", "2024-06-01"]), "symbol": ["A", "B"]})
    out = merge_fundamentals_pit(df, fundamentals_annual)

    assert out[out["symbol"] == "A"]["roe"].iloc[0] == pytest.approx(0.20)
    assert out[out["symbol"] == "B"]["roe"].iloc[0] == pytest.approx(0.99)


def test_merge_news_features_fills_no_news_days_with_zero():
    df = pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-01-02"]), "symbol": ["A", "A"]})
    news_daily = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"]),
            "symbol": ["A"],
            "news_count": [3],
            "sentiment_mean": [0.5],
            "sentiment_std": [0.1],
            "positive_news_ratio": [1.0],
            "negative_news_ratio": [0.0],
        }
    )
    out = merge_news_features(df, news_daily)

    assert out["news_count"].iloc[1] == 0.0
    assert out["sentiment_mean"].iloc[1] == 0.0
    assert out["news_count"].iloc[0] == 3


def test_merge_macro_and_regime_broadcasts_by_date_across_symbols():
    df = pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-01-01"]), "symbol": ["A", "B"]})
    macro_df = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"]), "sp500": [4000.0]})
    regime_df = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"]), "is_bull": [1.0]})

    out = merge_macro_and_regime(df, macro_df, regime_df)

    assert out["sp500"].tolist() == [4000.0, 4000.0]
    assert out["is_bull"].tolist() == [1.0, 1.0]
