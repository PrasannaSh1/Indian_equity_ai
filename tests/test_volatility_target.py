import numpy as np
import pandas as pd
import pytest

from src.risk.volatility_target import add_forward_volatility_target


def _prices(symbol, closes, start="2024-01-01"):
    dates = pd.date_range(start, periods=len(closes), freq="D")
    return pd.DataFrame({"date": dates, "symbol": symbol, "close": closes})


def test_forward_volatility_target_matches_hand_computed_window():
    # closes chosen so daily returns are easy to hand-verify:
    # closes:  100, 110, 121, 133.1, 146.41, 161.051, 177.1561
    # returns:      +10%  +10%   +10%    +10%     +10%     +10%  (all constant 10%)
    df = _prices("A", [100, 110, 121, 133.1, 146.41, 161.051, 177.1561])
    out = add_forward_volatility_target(df, horizon=2)

    # every return is exactly 10%, so the std of any 2-return window is 0
    assert out["forward_volatility_2d"].iloc[0] == pytest.approx(0.0, abs=1e-9)
    assert out["forward_volatility_2d"].iloc[4] == pytest.approx(0.0, abs=1e-9)
    # the last 2 rows have no full forward window of 2 future returns
    assert pd.isna(out["forward_volatility_2d"].iloc[-1])
    assert pd.isna(out["forward_volatility_2d"].iloc[-2])


def test_forward_volatility_target_uses_only_future_returns_not_past():
    # A volatile past (day0->day1 big drop) followed by a perfectly flat future
    # (day2 onward all +1%) -- if the window leaked past returns, row 1's target
    # would be nonzero; it must be 0 since it only looks at rows 2 and 3.
    df = _prices("A", [100.0, 50.0, 50.5, 51.005, 51.51505])
    out = add_forward_volatility_target(df, horizon=2)

    assert out["forward_volatility_2d"].iloc[1] == pytest.approx(0.0, abs=1e-9)


def test_forward_volatility_target_does_not_leak_across_symbols():
    df = pd.concat(
        [_prices("A", [100, 110, 121, 133.1]), _prices("B", [50, 45, 40.5, 36.45])],
        ignore_index=True,
    )
    out = add_forward_volatility_target(df, horizon=2)

    # last 2 rows of each symbol are NaN independently, not just the last 2 overall
    a_rows = out[out["symbol"] == "A"]
    b_rows = out[out["symbol"] == "B"]
    assert pd.isna(a_rows["forward_volatility_2d"].iloc[-1])
    assert pd.isna(b_rows["forward_volatility_2d"].iloc[-1])
    assert b_rows["forward_volatility_2d"].iloc[0] == pytest.approx(0.0, abs=1e-9)
