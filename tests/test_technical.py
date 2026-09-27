import pandas as pd
import pytest

from src.technical.momentum import adx, macd, roc, rsi, stochastic_oscillator
from src.technical.trend import ema, sma
from src.technical.volatility import atr, bollinger_bands
from src.technical.volume import obv, vwap, volume_ma, volume_ratio
from src.technical.score import technical_score


# ---- Trend ----------------------------------------------------------------


def test_sma_matches_hand_computed_rolling_mean():
    close = pd.Series([1, 2, 3, 4, 5])
    out = sma(close, window=3)

    assert out.iloc[:2].isna().all()
    assert out.iloc[2:].tolist() == pytest.approx([2.0, 3.0, 4.0])


def test_ema_matches_hand_computed_exponential_average():
    # span=3 -> alpha = 2/(span+1) = 0.5
    close = pd.Series([1, 2, 3, 4, 5])
    out = ema(close, span=3)

    assert out.tolist() == pytest.approx([1.0, 1.5, 2.25, 3.125, 4.0625])


# ---- Momentum ---------------------------------------------------------------


def test_rsi_is_100_for_a_strictly_increasing_series():
    close = pd.Series(range(1, 40))  # monotonic increase -> zero average loss
    out = rsi(close, period=14)

    assert out.iloc[-1] == pytest.approx(100.0)
    assert out.iloc[0] is not None and pd.isna(out.iloc[0])  # diff() leaves the first value NaN


def test_rsi_is_0_for_a_strictly_decreasing_series():
    close = pd.Series(range(40, 1, -1))  # monotonic decrease -> zero average gain
    out = rsi(close, period=14)

    assert out.iloc[-1] == pytest.approx(0.0)


def test_macd_is_zero_for_a_constant_price_series():
    close = pd.Series([100.0] * 40)
    out = macd(close)

    assert out["macd"].tolist() == pytest.approx([0.0] * 40)
    assert out["histogram"].tolist() == pytest.approx([0.0] * 40)


def test_roc_matches_hand_computed_percentage_change():
    close = pd.Series([100.0, 110.0, 121.0])
    out = roc(close, period=1)

    assert pd.isna(out.iloc[0])
    assert out.iloc[1] == pytest.approx(10.0)
    assert out.iloc[2] == pytest.approx(10.0)


def test_stochastic_oscillator_matches_hand_computed_percent_k():
    high = pd.Series([10, 12, 14])
    low = pd.Series([8, 9, 10])
    close = pd.Series([9, 11, 13])
    out = stochastic_oscillator(high, low, close, k_period=2, d_period=1)

    # idx1: lowest_low=min(8,9)=8, highest_high=max(10,12)=12 -> (11-8)/(12-8)*100 = 75
    # idx2: lowest_low=min(9,10)=9, highest_high=max(12,14)=14 -> (13-9)/(14-9)*100 = 80
    assert out["percent_k"].iloc[1] == pytest.approx(75.0)
    assert out["percent_k"].iloc[2] == pytest.approx(80.0)


def test_adx_is_high_for_a_strongly_trending_series():
    n = 40
    high = pd.Series(range(20, 20 + n), dtype=float)
    low = pd.Series(range(15, 15 + n), dtype=float)
    close = pd.Series(range(18, 18 + n), dtype=float)
    out = adx(high, low, close, period=14)

    assert out.iloc[-1] > 90  # a clean, uninterrupted uptrend should read near-maximal strength


# ---- Volatility -------------------------------------------------------------


def test_atr_equals_the_constant_true_range():
    # high-low is a constant 2 on every bar -> true range is 2 throughout, so ATR (a
    # smoothed average of a constant series) must also equal 2 throughout.
    high = pd.Series([10.0] * 6)
    low = pd.Series([8.0] * 6)
    close = pd.Series([9.0] * 6)
    out = atr(high, low, close, period=2)

    assert pd.isna(out.iloc[0])  # fewer than `period` observations available
    assert out.iloc[1:].tolist() == pytest.approx([2.0] * 5)


def test_bollinger_bands_collapse_to_the_mean_for_a_constant_series():
    close = pd.Series([5.0] * 25)
    out = bollinger_bands(close, window=20, num_std=2)

    tail = out.dropna()
    assert tail["bb_middle"].tolist() == pytest.approx([5.0] * len(tail))
    assert tail["bb_upper"].tolist() == pytest.approx([5.0] * len(tail))
    assert tail["bb_lower"].tolist() == pytest.approx([5.0] * len(tail))
    assert tail["bb_width"].tolist() == pytest.approx([0.0] * len(tail))


# ---- Volume -------------------------------------------------------------


def test_volume_ma_and_ratio_match_hand_computed_values():
    volume = pd.Series([10, 20, 30, 40])
    ma = volume_ma(volume, window=2)

    assert ma.tolist()[1:] == pytest.approx([15.0, 25.0, 35.0])
    ratio = volume_ratio(volume, window=2)
    assert ratio.iloc[3] == pytest.approx(40 / 35)


def test_obv_matches_hand_computed_cumulative_signed_volume():
    close = pd.Series([10, 11, 10, 12])
    volume = pd.Series([100, 200, 150, 300])
    out = obv(close, volume)

    assert out.tolist() == [0, 200, 50, 350]


def test_vwap_matches_hand_computed_value_for_constant_typical_price():
    high = pd.Series([10.0, 10.0])
    low = pd.Series([8.0, 8.0])
    close = pd.Series([9.0, 9.0])
    volume = pd.Series([100.0, 200.0])
    out = vwap(high, low, close, volume, window=2)

    assert pd.isna(out.iloc[0])
    assert out.iloc[1] == pytest.approx(9.0)


# ---- Composite score ---------------------------------------------------------------


def test_technical_score_matches_hand_computed_average_of_subscores():
    df = pd.DataFrame(
        {
            "close": [110.0],
            "sma_50": [100.0],
            "sma_200": [90.0],
            "rsi_14": [70.0],
            "volume_ratio": [1.5],
        }
    )
    # trend: close>sma50 (+50) and sma50>sma200 (+50) -> 100
    # momentum: 70
    # volume: 1.5*50 = 75
    # score = (100 + 70 + 75) / 3
    out = technical_score(df)

    assert out.iloc[0] == pytest.approx((100 + 70 + 75) / 3)


def test_technical_score_is_nan_when_sma_200_is_not_yet_available():
    # During warm-up, sma_50 > NaN evaluates to False in pandas rather than NaN,
    # so the score must be explicitly masked instead of silently defaulting.
    df = pd.DataFrame(
        {
            "close": [110.0],
            "sma_50": [100.0],
            "sma_200": [float("nan")],
            "rsi_14": [70.0],
            "volume_ratio": [1.5],
        }
    )
    out = technical_score(df)

    assert pd.isna(out.iloc[0])
