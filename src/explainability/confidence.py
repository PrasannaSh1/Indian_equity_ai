"""Prediction confidence, surfaced alongside every probability (project plan
Section 31/9: an explainable system should be able to say not just *what* it
predicts, but how sure it is).
"""

from __future__ import annotations

import pandas as pd

CONFIDENCE_LOW_MAX = 0.2
CONFIDENCE_MEDIUM_MAX = 0.5


def prediction_confidence(probability_up: pd.Series) -> pd.Series:
    """0 (coin-flip, probability=0.5) to 1 (certain, probability=0 or 1)."""
    return (probability_up - 0.5).abs() * 2


def confidence_label(confidence: pd.Series) -> pd.Series:
    """Buckets confidence into Low / Medium / High using fixed thresholds.

    Unlike the risk score (Phase 8), this 0-1 scale has a fixed, universe
    -agnostic meaning (distance from a coin flip), so absolute cutoffs are
    appropriate here rather than a learned/relative bucketing.
    """
    return pd.cut(
        confidence,
        bins=[-float("inf"), CONFIDENCE_LOW_MAX, CONFIDENCE_MEDIUM_MAX, float("inf")],
        labels=["Low", "Medium", "High"],
    )
