"""Data-sufficiency gating before ML inference (Phase 18).

Decides whether a dynamically-fetched company has enough history to trust a feature
vector, so the orchestrator can refuse to predict (INSUFFICIENT_DATA) rather than
feed the global model a mostly-NaN row. Never fabricates a score for missing data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.models.dataset import FEATURE_COLUMNS, build_feature_table

# sma_200 -- the slowest indicator FEATURE_COLUMNS depends on -- needs 200 trading
# days of warm-up, plus a further ~60-day lookback for the return/volatility
# features on top of that. This is a floor, not a guarantee: the real check below is
# the actual latest-row completeness test, not just this count.
MIN_HISTORY_TRADING_DAYS = 260


@dataclass
class EligibilityReport:
    history_days: int
    history_years: float
    feature_complete: bool
    ml_eligible: bool
    missing_features: list[str] = field(default_factory=list)
    reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "history_days": self.history_days,
            "history_years": round(self.history_years, 2),
            "feature_complete": self.feature_complete,
            "ml_eligible": self.ml_eligible,
            "missing_features": self.missing_features,
            "reason": self.reason,
        }


def assess_market_data_eligibility(indicator_df: pd.DataFrame) -> EligibilityReport:
    """`indicator_df` is a single-symbol frame that already has technical indicators
    (and technical_score) computed -- the same shape build_ml_dataset expects, e.g.
    src.technical.indicators.add_technical_indicators's output plus a technical_score
    column.
    """
    history_days = len(indicator_df)
    history_years = history_days / 252

    if history_days < MIN_HISTORY_TRADING_DAYS:
        return EligibilityReport(
            history_days=history_days,
            history_years=history_years,
            feature_complete=False,
            ml_eligible=False,
            reason=(
                f"Only {history_days} trading days of history (need >= "
                f"{MIN_HISTORY_TRADING_DAYS} for the slowest technical feature, SMA-200, to warm up)."
            ),
        )

    features = build_feature_table(indicator_df)
    latest = features.sort_values("date").iloc[-1]
    missing = [c for c in FEATURE_COLUMNS if pd.isna(latest.get(c))]

    return EligibilityReport(
        history_days=history_days,
        history_years=history_years,
        feature_complete=not missing,
        ml_eligible=not missing,
        missing_features=missing,
        reason=None if not missing else f"Latest row is missing: {missing}",
    )
