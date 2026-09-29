import pytest

from src.models.registry import load_all, load_latest, load_version, register_model


def _entry(version, training_date):
    return {
        "model_version": version,
        "model_name": "xgboost",
        "training_date": training_date,
        "feature_schema_version": "1.0",
    }


def test_register_and_load_all_roundtrip(tmp_path):
    register_model(_entry("v1", "2026-01-01T00:00:00+00:00"), model_dir=tmp_path)

    entries = load_all(model_dir=tmp_path)

    assert len(entries) == 1
    assert entries[0]["model_version"] == "v1"


def test_registering_same_version_replaces_not_duplicates(tmp_path):
    register_model(_entry("v1", "2026-01-01T00:00:00+00:00"), model_dir=tmp_path)
    register_model(_entry("v1", "2026-02-01T00:00:00+00:00"), model_dir=tmp_path)

    entries = load_all(model_dir=tmp_path)

    assert len(entries) == 1
    assert entries[0]["training_date"] == "2026-02-01T00:00:00+00:00"


def test_load_latest_picks_most_recent_training_date(tmp_path):
    register_model(_entry("v1", "2026-01-01T00:00:00+00:00"), model_dir=tmp_path)
    register_model(_entry("v2", "2026-03-01T00:00:00+00:00"), model_dir=tmp_path)

    latest = load_latest(model_dir=tmp_path)

    assert latest["model_version"] == "v2"


def test_load_latest_returns_none_when_registry_empty(tmp_path):
    assert load_latest(model_dir=tmp_path) is None


def test_load_version_raises_for_unknown_version(tmp_path):
    register_model(_entry("v1", "2026-01-01T00:00:00+00:00"), model_dir=tmp_path)

    with pytest.raises(ValueError):
        load_version("does-not-exist", model_dir=tmp_path)
