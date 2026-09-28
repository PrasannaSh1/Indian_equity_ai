import numpy as np
import pandas as pd
import pytest

from src.models.dataset import add_target, chronological_split


def _minimal_technical_df(symbol, closes, start="2024-01-01"):
    dates = pd.date_range(start, periods=len(closes), freq="D")
    n = len(closes)
    return pd.DataFrame(
        {
            "date": dates,
            "symbol": symbol,
            "open": closes,
            "high": [c + 1 for c in closes],
            "low": [c - 1 for c in closes],
            "close": closes,
            "adjusted_close": closes,
            "volume": [1000] * n,
        }
    )


def test_add_target_labels_up_and_down_days_correctly():
    df = _minimal_technical_df("A", [100, 110, 105, 105])
    out = add_target(df)

    # day0->day1 up (1), day1->day2 down (0), day2->day3 flat -> not up (0)
    assert out["next_day_direction"].iloc[:3].tolist() == [1.0, 0.0, 0.0]


def test_add_target_leaves_last_row_per_symbol_as_nan_not_zero():
    df = _minimal_technical_df("A", [100, 110, 105])
    out = add_target(df)

    assert pd.isna(out["next_day_direction"].iloc[-1])


def test_add_target_does_not_leak_across_symbols():
    df = pd.concat(
        [
            _minimal_technical_df("A", [100, 110]),
            _minimal_technical_df("B", [50, 40]),
        ],
        ignore_index=True,
    )
    out = add_target(df)

    a_last = out[out["symbol"] == "A"].iloc[-1]
    b_last = out[out["symbol"] == "B"].iloc[-1]
    assert pd.isna(a_last["next_day_direction"])
    assert pd.isna(b_last["next_day_direction"])


def test_chronological_split_has_no_date_overlap_and_correct_order():
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    df = pd.DataFrame({"date": dates, "value": range(100)})

    train, val, test = chronological_split(df, train_frac=0.7, val_frac=0.15)

    assert train["date"].max() < val["date"].min()
    assert val["date"].max() < test["date"].min()
    assert len(train) + len(val) + len(test) == len(df)
    assert len(train) == pytest.approx(70, abs=1)
    assert len(val) == pytest.approx(15, abs=1)
