import numpy as np
import pandas as pd
import pytest

from src.features.risk import add_drawdown, beta, correlation_matrix, max_drawdown, period_returns


def _prices(symbol, closes, start="2024-01-01", freq="D"):
    dates = pd.date_range(start, periods=len(closes), freq=freq)
    return pd.DataFrame({"date": dates, "symbol": symbol, "close": closes})


def test_add_drawdown_tracks_decline_from_peak():
    df = _prices("A", [100, 120, 90, 60, 80])
    out = add_drawdown(df)

    assert out["drawdown"].tolist() == pytest.approx([0.0, 0.0, -0.25, -0.5, -1 / 3], rel=1e-6)


def test_max_drawdown_is_the_most_negative_value_per_symbol():
    df = pd.concat(
        [
            _prices("A", [100, 120, 60]),  # drawdown -0.5
            _prices("B", [100, 90, 95]),  # drawdown -0.1
        ],
        ignore_index=True,
    )
    result = max_drawdown(df)

    assert result["A"] == pytest.approx(-0.5)
    assert result["B"] == pytest.approx(-0.1)


def test_beta_of_series_with_itself_is_one():
    benchmark = pd.Series([0.01, -0.02, 0.03, 0.0, 0.015])
    assert beta(benchmark, benchmark) == pytest.approx(1.0)


def test_beta_is_nan_when_no_overlap():
    a = pd.Series([0.01, 0.02], index=[0, 1])
    b = pd.Series([0.01, 0.02], index=[2, 3])
    assert np.isnan(beta(a, b))


def test_period_returns_computes_pct_change_on_resampled_close():
    df = _prices("A", [100, 105, 110, 120, 90, 95, 100, 130], freq="D")
    out = period_returns(df, freq="W-SUN")

    assert list(out.columns) == ["date", "symbol", "return"]
    assert out["symbol"].eq("A").all()


def test_correlation_matrix_of_perfectly_correlated_series_is_one():
    returns_wide = pd.DataFrame({"A": [0.01, 0.02, -0.01], "B": [0.02, 0.04, -0.02]})
    corr = correlation_matrix(returns_wide)

    assert corr.loc["A", "B"] == pytest.approx(1.0)
