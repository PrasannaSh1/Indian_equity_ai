import pandas as pd
import pytest

from src.entryexit.engine import compute_entry_exit, compute_entry_exit_validated


def test_compute_entry_exit_matches_hand_computed_zones_for_a_bullish_signal():
    close = pd.Series([100.0])
    atr = pd.Series([2.0])
    probability_up = pd.Series([0.6])

    out = compute_entry_exit(
        close, atr, probability_up,
        threshold=0.55, entry_buffer_pct=0.005, stop_atr_multiple=1.5, reward_risk_ratio=2.0,
    )

    assert out["has_signal"].iloc[0] == True  # noqa: E712
    assert out["entry_low"].iloc[0] == pytest.approx(100.0)
    assert out["entry_high"].iloc[0] == pytest.approx(100.5)
    assert out["stop"].iloc[0] == pytest.approx(97.0)  # 100 - 1.5*2
    assert out["target"].iloc[0] == pytest.approx(106.0)  # 100 + 2.0*(1.5*2)


def test_compute_entry_exit_suggests_no_position_when_not_bullish():
    close = pd.Series([100.0])
    atr = pd.Series([2.0])
    probability_up = pd.Series([0.4])

    out = compute_entry_exit(close, atr, probability_up, threshold=0.55)

    assert out["has_signal"].iloc[0] == False  # noqa: E712
    assert out[["entry_low", "entry_high", "stop", "target"]].iloc[0].isna().all()


def test_compute_entry_exit_wider_stop_gives_wider_target_for_the_same_reward_risk_ratio():
    close = pd.Series([100.0, 100.0])
    atr = pd.Series([1.0, 4.0])  # second row is much more volatile
    probability_up = pd.Series([0.9, 0.9])

    out = compute_entry_exit(close, atr, probability_up, threshold=0.55, stop_atr_multiple=1.0, reward_risk_ratio=2.0)

    assert out["stop"].iloc[1] < out["stop"].iloc[0]  # more volatile -> wider (lower) stop
    assert out["target"].iloc[1] > out["target"].iloc[0]  # and a further target, same R:R ratio


def test_compute_entry_exit_validated_marks_normal_output_as_valid():
    close = pd.Series([100.0])
    atr = pd.Series([2.0])
    probability_up = pd.Series([0.6])

    out = compute_entry_exit_validated(close, atr, probability_up, threshold=0.55)

    assert out["is_valid"].iloc[0] == True  # noqa: E712
    assert out["invalid_reason"].iloc[0] is None


def test_compute_entry_exit_validated_marks_no_signal_rows_as_valid():
    close = pd.Series([100.0])
    atr = pd.Series([2.0])
    probability_up = pd.Series([0.4])

    out = compute_entry_exit_validated(close, atr, probability_up, threshold=0.55)

    assert out["is_valid"].iloc[0] == True  # noqa: E712 -- no signal is valid by definition
