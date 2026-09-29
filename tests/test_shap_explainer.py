import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier

from src.explainability.shap_explainer import build_explainer, explain_predictions, top_factors


def _dominant_feature_model():
    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame(
        {
            "dominant": rng.normal(size=n),
            "noise_a": rng.normal(size=n),
            "noise_b": rng.normal(size=n),
        }
    )
    y = (X["dominant"] > 0).astype(int)
    model = LGBMClassifier(n_estimators=50, max_depth=3, verbose=-1, random_state=0)
    model.fit(X, y)
    return model, X


def test_explain_predictions_reconstructs_the_models_raw_margin():
    model, X = _dominant_feature_model()
    explainer = build_explainer(model)
    shap_df = explain_predictions(explainer, X.iloc[:10])

    reconstructed_margin = shap_df.sum(axis=1) + explainer.expected_value
    actual_margin = model.predict(X.iloc[:10], raw_score=True)

    assert reconstructed_margin.to_numpy() == pytest.approx(actual_margin, abs=1e-6)


def test_top_factors_identifies_the_genuinely_dominant_feature():
    model, X = _dominant_feature_model()
    explainer = build_explainer(model)

    # a strongly positive "dominant" value should show up as that row's top positive factor
    candidate_rows = X[X["dominant"] > 1]
    assert len(candidate_rows) > 0
    shap_df = explain_predictions(explainer, candidate_rows.iloc[:1])
    factors = top_factors(shap_df.iloc[0], n=3)

    assert factors["positive"].index[0] == "dominant"


def test_top_factors_splits_positive_and_negative_correctly():
    shap_row = pd.Series({"a": 0.5, "b": -0.3, "c": 0.1, "d": -0.05, "e": 0.0})
    factors = top_factors(shap_row, n=2)

    assert list(factors["positive"].index) == ["a", "c"]
    assert list(factors["negative"].index) == ["b", "d"]
    assert (factors["positive"] > 0).all()
    assert (factors["negative"] < 0).all()
