"""Composite 0-100 risk score and Low/Medium/High label.

Higher score = more risky. Combines four independently meaningful risk signals
into one equally weighted average -- the same documented-heuristic pattern as
the Technical Score (Phase 3) and Fundamental Score (Phase 4): simple, clipped,
absolute thresholds rather than a fitted model, chosen so a "typical" large-cap
stock lands near the middle of the range.
"""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = ("expected_volatility", "beta", "max_drawdown", "var_95")

LOW_RISK_MAX = 33.33
MEDIUM_RISK_MAX = 66.67


def compute_risk_score(df: pd.DataFrame) -> pd.Series:
    """Requires columns: expected_volatility (daily, fraction), beta (vs NIFTY 50),
    max_drawdown (<=0), var_95 (<=0, from historical_var).

    - volatility: 0% daily vol -> 0, 5%+ -> 100
    - beta: |beta|=0 -> 0, |beta|=2+ -> 100
    - drawdown: 0% historical drawdown -> 0, 60%+ -> 100
    - VaR: 0% 1-day VaR -> 0, 5%+ -> 100
    """
    vol_component = (df["expected_volatility"].clip(lower=0, upper=0.05) / 0.05) * 100
    beta_component = (df["beta"].abs().clip(lower=0, upper=2) / 2) * 100
    drawdown_component = (df["max_drawdown"].abs().clip(lower=0, upper=0.60) / 0.60) * 100
    var_component = (df["var_95"].abs().clip(lower=0, upper=0.05) / 0.05) * 100

    missing_inputs = df[list(REQUIRED_COLUMNS)].isna().any(axis=1)

    score = (vol_component + beta_component + drawdown_component + var_component) / 4
    score = score.clip(lower=0, upper=100)
    return score.where(~missing_inputs)


def risk_label(risk_score: pd.Series) -> pd.Series:
    """Buckets a 0-100 risk score into Low / Medium / High using fixed absolute
    thresholds. Only meaningful if the score's caps (see compute_risk_score) are
    well-calibrated to the universe being scored -- a narrow universe (e.g. only
    large-cap blue chips) can legitimately never reach the Low or High ends of an
    absolute scale calibrated for a broader market. See risk_label_relative for a
    universe-agnostic alternative.
    """
    return pd.cut(
        risk_score,
        bins=[-float("inf"), LOW_RISK_MAX, MEDIUM_RISK_MAX, float("inf")],
        labels=["Low", "Medium", "High"],
    )


def risk_label_relative(risk_score: pd.Series, reference_scores: pd.Series) -> pd.Series:
    """Buckets risk_score into Low/Medium/High using tertile cutoffs learned from
    `reference_scores` (e.g. the training split), not from risk_score itself, so
    the same cutoffs can be applied out-of-sample without the boundaries leaking
    information from the distribution being labeled. Always produces a meaningful
    3-way split regardless of how the absolute score happens to be calibrated for
    a given universe.
    """
    low_cutoff = reference_scores.quantile(1 / 3)
    high_cutoff = reference_scores.quantile(2 / 3)
    return pd.cut(
        risk_score,
        bins=[-float("inf"), low_cutoff, high_cutoff, float("inf")],
        labels=["Low", "Medium", "High"],
    )
