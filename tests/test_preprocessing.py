import pandas as pd

from src.preprocessing.clean import clean_ohlcv, validate_ohlcv


def _row(date, symbol="RELIANCE", open_=100, high=105, low=99, close=102, volume=1000):
    return {
        "date": pd.Timestamp(date),
        "symbol": symbol,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "adjusted_close": close,
        "volume": volume,
    }


def test_validate_ohlcv_flags_known_issues():
    df = pd.DataFrame(
        [
            _row("2024-01-01"),
            _row("2024-01-01"),  # duplicate
            _row("2024-01-02", close=-5),  # non-positive price
            _row("2024-01-03", volume=-10),  # negative volume
            _row("2024-01-04", high=50, low=60),  # high < low
        ]
    )
    issues = validate_ohlcv(df)
    assert issues["duplicate_rows"] == 1
    assert issues["non_positive_prices"] >= 1
    assert issues["negative_volume"] == 1
    assert issues["high_below_low"] == 1


def test_clean_ohlcv_removes_bad_rows():
    df = pd.DataFrame(
        [
            _row("2024-01-02"),
            _row("2024-01-01"),
            _row("2024-01-01"),  # duplicate of the row above
            _row("2024-01-03", close=0),  # invalid price
        ]
    )
    cleaned = clean_ohlcv(df)

    assert len(cleaned) == 2
    assert list(cleaned["date"]) == sorted(cleaned["date"])
    assert validate_ohlcv(cleaned) == {
        "missing_values": 0,
        "duplicate_rows": 0,
        "non_positive_prices": 0,
        "negative_volume": 0,
        "high_below_low": 0,
    }
