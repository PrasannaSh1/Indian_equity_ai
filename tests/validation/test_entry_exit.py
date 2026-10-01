import math

from src.validation.entry_exit import validate_long_setup


def test_validate_long_setup_flags_stop_above_entry_as_invalid():
    # Mirrors the reported bug: close ~2831, stop 2859 (above entry).
    result = validate_long_setup(stop=2859, entry_low=2831, entry_high=2845, target=3000)

    assert result.valid is False
    assert "Stop" in result.reason


def test_validate_long_setup_flags_target_below_entry_as_invalid():
    result = validate_long_setup(stop=2750, entry_low=2800, entry_high=2810, target=2790)

    assert result.valid is False
    assert "Target" in result.reason


def test_validate_long_setup_passes_correctly_ordered_levels():
    result = validate_long_setup(stop=2750, entry_low=2800, entry_high=2810, target=3000)

    assert result.valid is True
    assert result.reason is None


def test_validate_long_setup_no_signal_is_valid_by_definition():
    nan = math.nan
    result = validate_long_setup(stop=nan, entry_low=nan, entry_high=nan, target=nan)

    assert result.valid is True


def test_validate_long_setup_flags_partial_signal_as_invalid():
    result = validate_long_setup(stop=2750, entry_low=2800, entry_high=None, target=math.nan)

    assert result.valid is False
