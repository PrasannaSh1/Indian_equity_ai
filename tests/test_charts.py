import pandas as pd
import pytest

from app.charts import build_price_chart


def _history():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "open": [100.0] * 5, "high": [102.0] * 5, "low": [99.0] * 5, "close": [101.0] * 5,
            "volume": [1000.0] * 5,
            "sma_20": [100.0] * 5, "sma_50": [99.0] * 5,
            "bb_upper": [105.0] * 5, "bb_middle": [100.0] * 5, "bb_lower": [95.0] * 5,
            "rsi_14": [55.0] * 5, "macd": [0.1] * 5, "macd_signal": [0.05] * 5, "macd_histogram": [0.05] * 5,
        }
    )


def test_base_chart_has_candlestick_and_volume_only():
    fig = build_price_chart(_history())
    names = [trace.name for trace in fig.data]
    assert names == ["Price", "Volume"]


def test_sma_overlay_adds_one_line_trace_on_the_price_panel():
    fig = build_price_chart(_history(), overlays=["SMA 20"])
    names = [trace.name for trace in fig.data]
    assert names == ["Price", "SMA 20", "Volume"]


def test_bollinger_bands_overlay_adds_three_line_traces():
    fig = build_price_chart(_history(), overlays=["Bollinger Bands"])
    names = [trace.name for trace in fig.data]
    assert names == ["Price", "BB Upper", "BB Middle", "BB Lower", "Volume"]


def test_missing_indicator_columns_are_skipped_not_an_error():
    history = _history().drop(columns=["sma_50"])
    fig = build_price_chart(history, overlays=["SMA 20", "SMA 50"])
    names = [trace.name for trace in fig.data]
    assert "SMA 20" in names
    assert "SMA 50" not in names  # column absent -> silently skipped


def test_rsi_oscillator_adds_a_separate_subplot_row():
    fig = build_price_chart(_history(), oscillators=["RSI"])
    names = [trace.name for trace in fig.data]
    assert names == ["Price", "Volume", "RSI(14)"]
    rsi_trace = next(t for t in fig.data if t.name == "RSI(14)")
    assert rsi_trace.yaxis == "y3"  # its own subplot row, not sharing the price panel


def test_macd_oscillator_adds_three_traces():
    fig = build_price_chart(_history(), oscillators=["MACD"])
    names = [trace.name for trace in fig.data]
    assert names == ["Price", "Volume", "MACD", "MACD Signal", "MACD Histogram"]


def test_overlays_and_oscillators_combine():
    fig = build_price_chart(_history(), overlays=["SMA 20"], oscillators=["RSI", "MACD"])
    names = [trace.name for trace in fig.data]
    assert names == ["Price", "SMA 20", "Volume", "RSI(14)", "MACD", "MACD Signal", "MACD Histogram"]
