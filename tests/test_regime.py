import pandas as pd
import pytest

from src.regime.classifier import classify_market_regime


def test_classify_market_regime_trend_matches_hand_computed_pct_change():
    nifty = pd.Series([100.0, 105.0, 95.0, 90.0, 120.0])
    vix = pd.Series([10.0] * 5)  # irrelevant to this assertion

    out = classify_market_regime(nifty, vix, trend_window=2, min_vol_history=100)

    assert pd.isna(out["is_bull"].iloc[0])
    assert pd.isna(out["is_bull"].iloc[1])
    assert out["is_bull"].iloc[2] == 0.0  # (95-100)/100 = -5% -> bear
    assert out["is_bull"].iloc[3] == 0.0  # (90-105)/105 = -14.3% -> bear
    assert out["is_bull"].iloc[4] == 1.0  # (120-95)/95 = +26.3% -> bull


def test_classify_market_regime_volatility_uses_expanding_median_not_full_sample():
    nifty = pd.Series([100.0] * 6)  # irrelevant to this assertion
    vix = pd.Series([10.0, 20.0, 12.0, 30.0, 8.0, 15.0])

    out = classify_market_regime(nifty, vix, trend_window=1, min_vol_history=3)

    assert pd.isna(out["is_high_vol"].iloc[0])
    assert pd.isna(out["is_high_vol"].iloc[1])
    # expanding median at each point: [10,20,12]->12, [..,30]->16, [..,8]->12, [..,15]->13.5
    assert out["is_high_vol"].iloc[2] == 0.0  # 12 > 12 is False
    assert out["is_high_vol"].iloc[3] == 1.0  # 30 > 16
    assert out["is_high_vol"].iloc[4] == 0.0  # 8 > 12 is False
    assert out["is_high_vol"].iloc[5] == 1.0  # 15 > 13.5
