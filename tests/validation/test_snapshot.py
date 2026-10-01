from datetime import date

import pytest

from src.validation.snapshot import assert_single_snapshot, resolve_snapshot


def test_resolve_snapshot_takes_min_of_available_source_dates():
    result = resolve_snapshot(
        {
            "prices_daily": date(2026, 9, 28),
            "technical_features": date(2026, 9, 28),
            "model_predictions": date(2026, 9, 21),
            "entry_exit": date(2026, 9, 21),
        }
    )

    assert result.as_of == date(2026, 9, 21)
    assert result.is_fully_aligned is False
    assert result.lagging_sources == {"prices_daily": 7, "technical_features": 7}


def test_resolve_snapshot_ignores_sources_with_no_data():
    result = resolve_snapshot({"prices_daily": date(2026, 9, 28), "news": None})

    assert result.as_of == date(2026, 9, 28)
    assert "news" not in result.per_source_dates
    assert result.is_fully_aligned is True


def test_resolve_snapshot_fully_aligned_when_all_sources_match():
    result = resolve_snapshot({"a": date(2026, 9, 28), "b": date(2026, 9, 28)})

    assert result.is_fully_aligned is True
    assert result.lagging_sources == {}


def test_resolve_snapshot_raises_when_no_source_has_data():
    with pytest.raises(ValueError):
        resolve_snapshot({"prices_daily": None, "news": None})


def test_assert_single_snapshot_raises_on_divergence():
    with pytest.raises(AssertionError):
        assert_single_snapshot({"technical": date(2026, 9, 28), "prediction": date(2026, 9, 21)})


def test_assert_single_snapshot_passes_when_consistent():
    assert_single_snapshot({"technical": date(2026, 9, 28), "prediction": date(2026, 9, 28)})
