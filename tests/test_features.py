import numpy as np
import pandas as pd
import pytest

from src.features.returns import add_returns, add_rolling_volatility


def _prices(symbol, closes, start="2024-01-01"):
    dates = pd.date_range(start, periods=len(closes), freq="D")
    return pd.DataFrame({"date": dates, "symbol": symbol, "close": closes})


def test_add_returns_computes_pct_change_per_symbol():
    df = pd.concat(
        [
            _prices("A", [100, 110, 121]),
            _prices("B", [50, 45, 40]),
        ],
        ignore_index=True,
    )
    out = add_returns(df, periods=(1,))

    a = out[out["symbol"] == "A"].sort_values("date")
    assert np.isnan(a["return_1d"].iloc[0])
    assert a["return_1d"].iloc[1] == pytest.approx(0.10)
    assert a["return_1d"].iloc[2] == pytest.approx(0.10)


def test_add_rolling_volatility_is_nan_before_window_is_full():
    df = _prices("A", [100, 101, 99, 102, 98, 103])
    out = add_rolling_volatility(df, windows=(3,))

    assert out["volatility_3d"].iloc[:3].isna().all()
    assert out["volatility_3d"].iloc[3:].notna().all()
