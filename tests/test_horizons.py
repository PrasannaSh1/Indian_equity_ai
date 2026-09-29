import pandas as pd
import pytest

from src.models.horizons import add_horizon_target


def _prices(symbol, closes, start="2024-01-01"):
    dates = pd.date_range(start, periods=len(closes), freq="D")
    return pd.DataFrame({"date": dates, "symbol": symbol, "close": closes})


def test_add_horizon_target_matches_hand_computed_forward_direction():
    # closes: 100, 105, 95, 90, 120, 80 ; horizon=3
    # row0: close[0+3]=90 < 100 -> 0 (down)
    # row1: close[1+3]=120 > 105 -> 1 (up)
    # row2: close[2+3]=80 < 95 -> 0 (down)
    df = _prices("A", [100, 105, 95, 90, 120, 80])
    out = add_horizon_target(df, horizon_days=3, target_col="target_3d")

    assert out["target_3d"].iloc[:3].tolist() == [0.0, 1.0, 0.0]


def test_add_horizon_target_leaves_rows_without_a_full_window_as_nan_not_zero():
    df = _prices("A", [100, 105, 95, 90, 120, 80])
    out = add_horizon_target(df, horizon_days=3, target_col="target_3d")

    # last 3 rows have no close[t+3] available
    assert out["target_3d"].iloc[-3:].isna().all()


def test_add_horizon_target_does_not_leak_across_symbols():
    df = pd.concat(
        [_prices("A", [100, 110, 121, 133]), _prices("B", [50, 45, 40, 36])],
        ignore_index=True,
    )
    out = add_horizon_target(df, horizon_days=2, target_col="target_2d")

    a_rows = out[out["symbol"] == "A"]
    b_rows = out[out["symbol"] == "B"]
    assert pd.isna(a_rows["target_2d"].iloc[-1])
    assert pd.isna(b_rows["target_2d"].iloc[-1])
    assert not pd.isna(a_rows["target_2d"].iloc[0])
