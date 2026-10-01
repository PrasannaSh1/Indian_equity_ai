import pandas as pd

from src.security.resolver import ResolvedSecurity
from src.services.context import build_analysis_context
from src.services.lineage import build_lineage_panel


def _context(news_provider=None):
    resolved = ResolvedSecurity(symbol="TCS", name="Tata Consultancy Services", in_training_universe=True, resolved_by="ticker")
    latest_row = pd.Series({"date": pd.Timestamp("2026-09-28"), "close": 2070.70})
    news_meta = {"provider": news_provider, "retrieved_at": "2026-09-28T10:00:00+00:00"} if news_provider else None
    return build_analysis_context(
        identifier="TCS",
        resolved=resolved,
        latest_row=latest_row,
        entry={"model_version": "global_v1", "training_data_cutoff_date": "2026-08-31", "feature_schema_version": "1.0"},
        quantile_entry={"model_version": "quantile_v1", "training_data_cutoff_date": "2026-08-31"},
        fundamentals_block={
            "available": True,
            "balance_sheet_basis": {"fiscal_year": "2026-03-31"},
            "valuation_basis": {"as_of": "2026-09-28"},
        },
        news_meta=news_meta,
    )


def test_build_lineage_panel_has_one_row_per_category():
    panel = build_lineage_panel(_context(news_provider="yahoo_finance"), model_entry=None, quantile_entry=None)

    categories = {row["category"] for row in panel}
    assert categories == {
        "market_data",
        "fundamentals_balance_sheet",
        "fundamentals_valuation",
        "news",
        "model_classifier",
        "model_quantile",
    }


def test_build_lineage_panel_separates_training_cutoff_from_market_data_as_of():
    context = _context(news_provider="yahoo_finance")
    model_entry = {"training_data_cutoff_date": "2026-08-31"}

    panel = build_lineage_panel(context, model_entry=model_entry, quantile_entry=None)

    market_row = next(r for r in panel if r["category"] == "market_data")
    model_row = next(r for r in panel if r["category"] == "model_classifier")

    assert market_row["as_of_or_period"] == "2026-09-28"
    assert "2026-08-31" in model_row["as_of_or_period"]
    assert market_row["as_of_or_period"] != model_row["as_of_or_period"]


def test_build_lineage_panel_reports_unavailable_when_news_missing():
    panel = build_lineage_panel(_context(news_provider=None), model_entry=None, quantile_entry=None)

    news_row = next(r for r in panel if r["category"] == "news")
    assert news_row["source"] == "unavailable"
    assert news_row["as_of_or_period"] == "unavailable"


def test_build_lineage_panel_every_row_has_a_tier_label():
    panel = build_lineage_panel(_context(news_provider="yahoo_finance"), model_entry=None, quantile_entry=None)

    assert all(row["tier_label"] for row in panel)
