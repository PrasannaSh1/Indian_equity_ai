import pandas as pd
import pytest

from src.risk.var_es import historical_expected_shortfall, historical_var, rolling_var_es


def test_historical_var_matches_pandas_quantile_interpolation():
    returns = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # confidence=0.9 -> quantile(0.1); n=10, position=(10-1)*0.1=0.9 -> interpolate 1..2
    assert historical_var(returns, confidence=0.9) == pytest.approx(1.9)


def test_historical_expected_shortfall_is_the_mean_of_the_tail_at_or_below_var():
    returns = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # VaR_90 = 1.9 (see above); only the value 1 is <= 1.9
    assert historical_expected_shortfall(returns, confidence=0.9) == pytest.approx(1.0)


def _prices(symbol, closes, start="2024-01-01"):
    dates = pd.date_range(start, periods=len(closes), freq="D")
    return pd.DataFrame({"date": dates, "symbol": symbol, "close": closes})


def test_rolling_var_es_matches_hand_computed_median_based_values():
    # closes chosen so return_1d comes out to exactly 1,2,3,4,5,6 (%, as fractions)
    closes = [100, 101, 103.02, 106.1106, 110.435, 115.957]
    df = _prices("A", closes)

    out = rolling_var_es(df, window=3, confidence=0.5)  # confidence=0.5 -> VaR = rolling median

    returns = out["return_1d"].round(2).tolist()
    assert returns[1:] == pytest.approx([0.01, 0.02, 0.03, 0.04, 0.05], abs=1e-2)

    # window [r1,r2,r3]=[.01,.02,.03] -> median .02; window [r2,r3,r4]=[.02,.03,.04] -> median .03
    assert out["var_50"].iloc[3] == pytest.approx(0.02, abs=1e-3)
    assert out["var_50"].iloc[4] == pytest.approx(0.03, abs=1e-3)
    # ES = mean of the tail <= median: for [.01,.02,.03] median=.02, tail=[.01,.02] -> mean=.015
    assert out["expected_shortfall_50"].iloc[3] == pytest.approx(0.015, abs=1e-3)


def test_rolling_var_es_does_not_leak_across_symbols():
    df = pd.concat(
        [
            _prices("A", [100, 101, 103.02, 106.1106, 110.435]),
            _prices("B", [50, 45, 40.5, 36.45, 32.805]),
        ],
        ignore_index=True,
    )
    out = rolling_var_es(df, window=3, confidence=0.5)

    b_rows = out[out["symbol"] == "B"].reset_index(drop=True)
    assert pd.isna(b_rows["var_50"].iloc[0])
    assert pd.isna(b_rows["var_50"].iloc[1])
    assert b_rows["var_50"].iloc[3] < 0  # B is a steady decline, its rolling VaR should be negative
