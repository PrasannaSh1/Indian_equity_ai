from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.models.dataset import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION
from src.models.predict_quantiles import NoTrainedQuantileModelError, load_quantile_models, predict_price_quantiles
from src.models.registry import QUANTILE_REGISTRY_FILENAME, register_model


def _feature_row():
    return pd.DataFrame([{col: 0.1 for col in FEATURE_COLUMNS}])


class _FakeQuantileModel:
    def __init__(self, return_value):
        self._return_value = return_value

    def predict(self, X):
        return np.full(len(X), self._return_value)


def test_load_quantile_models_raises_when_none_registered(tmp_path):
    with pytest.raises(NoTrainedQuantileModelError):
        load_quantile_models(model_dir=tmp_path)


def test_load_quantile_models_raises_on_feature_schema_mismatch(tmp_path):
    register_model(
        {
            "model_version": "quantile_v1",
            "quantiles": [0.1, 0.5, 0.9],
            "model_paths": {},
            "training_date": "2026-01-01T00:00:00+00:00",
            "feature_schema_version": "0.9-stale",
        },
        model_dir=tmp_path,
        filename=QUANTILE_REGISTRY_FILENAME,
    )

    with pytest.raises(NoTrainedQuantileModelError):
        load_quantile_models(model_dir=tmp_path)


def test_predict_price_quantiles_matches_hand_computed_range():
    # q10 predicts a -5% return, q50 flat, q90 predicts +5% -- close=100 should give
    # a [95, 100, 105] range after return_quantiles_to_price_range's conversion.
    models = {
        0.1: _FakeQuantileModel(-0.05),
        0.5: _FakeQuantileModel(0.0),
        0.9: _FakeQuantileModel(0.05),
    }
    features = _feature_row()
    close = pd.Series([100.0], index=features.index)

    result = predict_price_quantiles(features, close, models)

    assert result["q10"].iloc[0] == pytest.approx(95.0)
    assert result["q50"].iloc[0] == pytest.approx(100.0)
    assert result["q90"].iloc[0] == pytest.approx(105.0)


def test_predict_price_quantiles_raises_on_missing_features():
    models = {0.1: _FakeQuantileModel(-0.05), 0.5: _FakeQuantileModel(0.0), 0.9: _FakeQuantileModel(0.05)}
    incomplete = pd.DataFrame([{FEATURE_COLUMNS[0]: 0.1}])
    close = pd.Series([100.0], index=incomplete.index)

    with pytest.raises(ValueError):
        predict_price_quantiles(incomplete, close, models)
