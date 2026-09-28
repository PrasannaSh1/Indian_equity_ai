import pandas as pd
import pytest

from src.macro.features import add_macro_returns


def test_add_macro_returns_matches_hand_computed_pct_change():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3, freq="D"),
            "india_vix": [10.0, 11.0, 9.9],
            "usdinr": [80.0, 81.6, 81.6],
            "brent_crude": [70.0, 70.0, 70.0],
            "us10y_yield": [4.0, 4.0, 4.0],
            "sp500": [4000.0, 4000.0, 4000.0],
        }
    )
    out = add_macro_returns(df)

    assert pd.isna(out["india_vix_return_1d"].iloc[0])
    assert out["india_vix_return_1d"].iloc[1] == pytest.approx(0.10)
    assert out["india_vix_return_1d"].iloc[2] == pytest.approx(-0.10)
    assert out["usdinr_return_1d"].iloc[1] == pytest.approx(0.02)
