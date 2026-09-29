import pandas as pd
import pytest

from src.risk.engine import compute_risk_score, risk_label, risk_label_relative


def test_compute_risk_score_matches_hand_computed_average():
    df = pd.DataFrame(
        {
            "expected_volatility": [0.025],  # -> 50
            "beta": [1.0],  # -> 50
            "max_drawdown": [-0.30],  # -> 50
            "var_95": [-0.025],  # -> 50
        }
    )
    out = compute_risk_score(df)
    assert out.iloc[0] == pytest.approx(50.0)


def test_compute_risk_score_uses_absolute_value_for_signed_inputs():
    df = pd.DataFrame({"expected_volatility": [0.05], "beta": [-2.0], "max_drawdown": [-0.60], "var_95": [-0.05]})
    out = compute_risk_score(df)
    assert out.iloc[0] == pytest.approx(100.0)


def test_compute_risk_score_is_nan_when_a_required_input_is_missing():
    df = pd.DataFrame(
        {"expected_volatility": [0.02], "beta": [1.0], "max_drawdown": [-0.1], "var_95": [float("nan")]}
    )
    out = compute_risk_score(df)
    assert pd.isna(out.iloc[0])


def test_risk_label_buckets_correctly():
    scores = pd.Series([10.0, 50.0, 90.0])
    labels = risk_label(scores)
    assert labels.tolist() == ["Low", "Medium", "High"]


def test_risk_label_boundary_is_inclusive_on_the_lower_side():
    scores = pd.Series([33.33, 33.34, 66.67, 66.68])
    labels = risk_label(scores)
    assert labels.tolist() == ["Low", "Medium", "Medium", "High"]


def test_risk_label_relative_spans_all_buckets_for_a_tightly_clustered_universe():
    # A narrow universe where every absolute score sits in the "Medium" band --
    # risk_label would collapse to all-Medium; risk_label_relative must still
    # produce a meaningful spread using tertiles of the reference distribution.
    reference_scores = pd.Series([40.0, 42.0, 44.0, 46.0, 48.0, 50.0])
    scores_to_label = pd.Series([40.0, 45.0, 50.0])

    labels = risk_label_relative(scores_to_label, reference_scores)

    assert labels.tolist() == ["Low", "Medium", "High"]
