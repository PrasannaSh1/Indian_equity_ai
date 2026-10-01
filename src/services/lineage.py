"""Data & Model Lineage panel (website audit Sections 32-33): one row per data
category showing source, trust tier, and as-of date/period. Reuses
src.rag.documents' existing Tier 1-4 / model_output source-reliability system
rather than inventing a new one.
"""

from __future__ import annotations

from src.rag.documents import TIER_FINANCIAL_WEBSITE, TIER_LABELS, TIER_MODEL_OUTPUT, provider_tier
from src.services.context import AnalysisContext


def _model_entry_label(entry: dict | None) -> str:
    if not entry or not entry.get("training_data_cutoff_date"):
        return "unavailable"
    return f"training cutoff {entry['training_data_cutoff_date']}"


def build_lineage_panel(context: AnalysisContext, model_entry: dict | None, quantile_entry: dict | None) -> list[dict]:
    news_tier = provider_tier(context.news_provider) if context.news_provider else TIER_FINANCIAL_WEBSITE

    return [
        {
            "category": "market_data",
            "source": context.sources.get("market_data", "unavailable"),
            "tier": TIER_FINANCIAL_WEBSITE,
            "tier_label": TIER_LABELS[TIER_FINANCIAL_WEBSITE],
            "as_of_or_period": context.market_data_as_of,
        },
        {
            "category": "fundamentals_balance_sheet",
            "source": context.sources.get("fundamentals", "unavailable"),
            "tier": TIER_FINANCIAL_WEBSITE,
            "tier_label": TIER_LABELS[TIER_FINANCIAL_WEBSITE],
            "as_of_or_period": context.fundamentals_fiscal_year or "unavailable",
        },
        {
            "category": "fundamentals_valuation",
            "source": context.sources.get("fundamentals", "unavailable"),
            "tier": TIER_FINANCIAL_WEBSITE,
            "tier_label": TIER_LABELS[TIER_FINANCIAL_WEBSITE],
            "as_of_or_period": context.fundamentals_valuation_as_of or "unavailable",
        },
        {
            "category": "news",
            "source": context.news_provider or "unavailable",
            "tier": news_tier,
            "tier_label": TIER_LABELS[news_tier],
            "as_of_or_period": context.news_retrieved_at or "unavailable",
        },
        {
            "category": "model_classifier",
            "source": "This project's own ML pipeline (global cross-company classifier)",
            "tier": TIER_MODEL_OUTPUT,
            "tier_label": TIER_LABELS[TIER_MODEL_OUTPUT],
            "as_of_or_period": _model_entry_label(model_entry),
        },
        {
            "category": "model_quantile",
            "source": "This project's own ML pipeline (quantile-regression price range)",
            "tier": TIER_MODEL_OUTPUT,
            "tier_label": TIER_LABELS[TIER_MODEL_OUTPUT],
            "as_of_or_period": _model_entry_label(quantile_entry),
        },
    ]
