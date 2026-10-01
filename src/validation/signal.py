"""Confidence-gated signal classification (website audit, Sections 18-21).

Root cause this fixes: the dashboard picked "Bullish"/"Bearish" purely from
probability_up > 0.5, ignoring model confidence entirely, so a 57.9% probability
with confidence 0.11 (a near coin-flip) was presented with the same unqualified
"Bullish" label as a high-confidence call.

Note that in this codebase confidence is NOT an independent calibration measure --
src.explainability.confidence.prediction_confidence() defines it as
abs(probability_up - 0.5) * 2, a deterministic function of probability_up itself.
So this classifier does not (and cannot) combine two independent signals; it reuses
confidence purely as a distance-from-coin-flip gate on top of the same probability,
using the SAME thresholds src.explainability.confidence already established
(CONFIDENCE_LOW_MAX, CONFIDENCE_MEDIUM_MAX) rather than inventing new ones.

These thresholds are a reasonable starting policy, NOT validated against real
calibration data -- this project's own documented finding (quantile forecasts:
~64% empirical coverage vs an ~80% target, see SUMMARY.md) is a specific reason for
caution about overstating precision here. A future walk-forward calibration study
(reliability diagrams / Brier score / isotonic or Platt calibration) should revisit
these cutoffs before they're treated as anything more than a labeling heuristic.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.explainability.confidence import CONFIDENCE_LOW_MAX, CONFIDENCE_MEDIUM_MAX


@dataclass(frozen=True)
class SignalClassification:
    label: str  # "Strong Bullish" | "Bullish" | "Neutral" | "Bearish" | "Strong Bearish" | "Low Confidence"
    probability_up: float
    confidence: float


def classify_signal(probability_up: float, confidence: float) -> SignalClassification:
    """Confidence below CONFIDENCE_LOW_MAX overrides direction entirely -- a
    near-coin-flip probability is reported as "Low Confidence", never dressed up
    as directional. Otherwise, direction comes from probability_up (>0.5 bullish,
    <0.5 bearish, ==0.5 neutral -- though ==0.5 implies confidence==0, already
    caught above), with a "Strong" prefix once confidence reaches
    CONFIDENCE_MEDIUM_MAX (the same cutoff confidence_label() calls "High").
    """
    if confidence < CONFIDENCE_LOW_MAX:
        label = "Low Confidence"
    elif probability_up > 0.5:
        label = "Strong Bullish" if confidence >= CONFIDENCE_MEDIUM_MAX else "Bullish"
    elif probability_up < 0.5:
        label = "Strong Bearish" if confidence >= CONFIDENCE_MEDIUM_MAX else "Bearish"
    else:
        label = "Neutral"

    return SignalClassification(label=label, probability_up=probability_up, confidence=confidence)
