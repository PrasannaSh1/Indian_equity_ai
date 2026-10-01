import pandas as pd

from src.security.resolver import ResolvedSecurity
from src.services.context import build_analysis_context


def _resolved(in_training_universe=True):
    return ResolvedSecurity(
        symbol="TCS", name="Tata Consultancy Services", in_training_universe=in_training_universe, resolved_by="ticker"
    )


def _latest_row(date="2026-09-28", close=2070.70):
    return pd.Series({"date": pd.Timestamp(date), "close": close})


def test_build_analysis_context_captures_market_and_model_fields():
    entry = {
        "model_version": "global_v1",
        "training_data_cutoff_date": "2026-08-31",
        "horizon_days": 1,
        "feature_schema_version": "1.0",
    }
    quantile_entry = {"model_version": "quantile_v1", "training_data_cutoff_date": "2026-08-31"}
    fundamentals_block = {
        "available": True,
        "balance_sheet_basis": {"fiscal_year": "2026-03-31"},
        "valuation_basis": {"as_of": "2026-09-28"},
    }
    news_meta = {"provider": "yahoo_finance", "retrieved_at": "2026-09-28T10:00:00+00:00"}

    context = build_analysis_context(
        identifier="TCS",
        resolved=_resolved(),
        latest_row=_latest_row(),
        entry=entry,
        quantile_entry=quantile_entry,
        fundamentals_block=fundamentals_block,
        news_meta=news_meta,
    )

    assert context.symbol == "TCS"
    assert context.in_training_universe is True
    assert context.market_data_as_of == "2026-09-28"
    assert context.latest_close == 2070.70
    assert context.classifier_model_version == "global_v1"
    assert context.classifier_training_data_cutoff == "2026-08-31"
    assert context.served_horizon_days == 1
    assert context.quantile_model_version == "quantile_v1"
    assert context.fundamentals_fiscal_year == "2026-03-31"
    assert context.fundamentals_valuation_as_of == "2026-09-28"
    assert context.news_provider == "yahoo_finance"


def test_build_analysis_context_handles_missing_optional_data_gracefully():
    context = build_analysis_context(
        identifier="DIXON",
        resolved=_resolved(in_training_universe=False),
        latest_row=_latest_row(),
        entry=None,
        quantile_entry=None,
        fundamentals_block={"available": False},
        news_meta=None,
    )

    assert context.in_training_universe is False
    assert context.classifier_model_version is None
    assert context.classifier_training_data_cutoff is None
    assert context.quantile_model_version is None
    assert context.fundamentals_fiscal_year is None
    assert context.news_provider is None
    assert context.feature_schema_version  # falls back to the current code's constant, never empty


def test_analysis_context_to_dict_is_json_shaped():
    context = build_analysis_context(
        identifier="TCS",
        resolved=_resolved(),
        latest_row=_latest_row(),
        entry=None,
        quantile_entry=None,
        fundamentals_block={"available": False},
        news_meta=None,
    )

    payload = context.to_dict()

    assert payload["symbol"] == "TCS"
    assert isinstance(payload["sources"], dict)
