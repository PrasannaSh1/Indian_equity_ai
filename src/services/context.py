"""Explicit single-analysis-snapshot summary for the live orchestrator (website
audit Sections 2, 34): makes the already-correct implicit pattern in
analyze_company() -- one fetch, one latest_row, shared by every downstream block
-- inspectable as one object, and explicitly separates the classifier's training
cutoff from the live market-data date it's being applied to.

Built once, at the end of analyze_company(), from values already in scope --
not threaded as a mutable object through every internal helper. The helpers
already correctly derive every date from the one shared indicator_df/latest_row
fetch; this is a reporting/transparency layer on top of that, not a different
data-flow architecture.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import pandas as pd

from src.models.dataset import FEATURE_SCHEMA_VERSION
from src.security.resolver import ResolvedSecurity


@dataclass(frozen=True)
class AnalysisContext:
    identifier: str
    symbol: str
    company_name: str | None
    in_training_universe: bool
    resolved_by: str
    analysis_as_of: str
    market_data_as_of: str
    fundamentals_fiscal_year: str | None
    fundamentals_valuation_as_of: str | None
    news_retrieved_at: str | None
    news_provider: str | None
    latest_close: float | None
    classifier_model_version: str | None
    classifier_training_data_cutoff: str | None
    quantile_model_version: str | None
    served_horizon_days: int | None
    feature_schema_version: str
    sources: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def build_analysis_context(
    *,
    identifier: str,
    resolved: ResolvedSecurity,
    latest_row: pd.Series,
    entry: dict | None,
    quantile_entry: dict | None,
    fundamentals_block: dict,
    news_meta: dict | None,
) -> AnalysisContext:
    news_meta = news_meta or {}
    balance_sheet = fundamentals_block.get("balance_sheet_basis") or {}
    valuation = fundamentals_block.get("valuation_basis") or {}
    news_provider = news_meta.get("provider")

    close = latest_row.get("close")
    return AnalysisContext(
        identifier=identifier,
        symbol=resolved.symbol,
        company_name=resolved.name,
        in_training_universe=resolved.in_training_universe,
        resolved_by=resolved.resolved_by,
        analysis_as_of=datetime.now(timezone.utc).isoformat(),
        market_data_as_of=str(pd.Timestamp(latest_row["date"]).date()),
        fundamentals_fiscal_year=balance_sheet.get("fiscal_year") if fundamentals_block.get("available") else None,
        fundamentals_valuation_as_of=valuation.get("as_of") if fundamentals_block.get("available") else None,
        news_retrieved_at=news_meta.get("retrieved_at"),
        news_provider=news_provider,
        latest_close=float(close) if close is not None and not pd.isna(close) else None,
        classifier_model_version=entry.get("model_version") if entry else None,
        classifier_training_data_cutoff=entry.get("training_data_cutoff_date") if entry else None,
        quantile_model_version=quantile_entry.get("model_version") if quantile_entry else None,
        served_horizon_days=entry.get("horizon_days") if entry else None,
        feature_schema_version=(entry or {}).get("feature_schema_version", FEATURE_SCHEMA_VERSION),
        sources={
            "market_data": "Yahoo Finance (relayed NSE EOD data)",
            "fundamentals": "Yahoo Finance (aggregated financial data)",
            "news": news_provider or "unavailable",
        },
    )
