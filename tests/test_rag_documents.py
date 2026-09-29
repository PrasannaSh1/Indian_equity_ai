import pandas as pd
import pytest

from src.rag.documents import (
    TIER_FINANCIAL_WEBSITE,
    TIER_MAJOR_NEWS,
    TIER_MODEL_OUTPUT,
    build_explanation_documents,
    build_fundamentals_documents,
    build_news_documents,
    build_risk_documents,
    build_technical_documents,
    provider_tier,
)


def test_provider_tier_recognizes_major_wire_services_case_insensitively():
    assert provider_tier("Reuters") == TIER_MAJOR_NEWS
    assert provider_tier("BLOOMBERG") == TIER_MAJOR_NEWS


def test_provider_tier_defaults_unknown_providers_to_financial_website():
    assert provider_tier("Simply Wall St.") == TIER_FINANCIAL_WEBSITE
    assert provider_tier("Some Random Blog") == TIER_FINANCIAL_WEBSITE


def test_build_fundamentals_documents_are_tagged_tier_3_not_tier_1():
    snapshot = pd.DataFrame(
        {"roe": [0.15], "debt_to_equity": [0.5], "revenue_growth_yoy": [0.1], "pe_ratio": [20.0],
         "fundamental_score": [60.0]},
        index=["RELIANCE"],
    )
    docs = build_fundamentals_documents(snapshot)

    assert len(docs) == 1
    assert docs[0]["symbol"] == "RELIANCE"
    assert docs[0]["tier"] == TIER_FINANCIAL_WEBSITE  # honestly tiered: Yahoo-sourced, not a direct filing
    assert "RELIANCE" in docs[0]["text"]
    assert "60" in docs[0]["text"]


def test_build_technical_documents_reflects_actual_sma_relationship():
    df = pd.DataFrame(
        {
            "symbol": ["TCS"],
            "date": [pd.Timestamp("2024-01-01")],
            "close": [100.0],
            "rsi_14": [55.0],
            "macd_histogram": [0.5],
            "sma_50": [90.0],  # close is ABOVE sma_50
            "sma_200": [110.0],  # close is BELOW sma_200
            "technical_score": [50.0],
        }
    )
    docs = build_technical_documents(df)

    text = docs[0]["text"]
    assert "above" in text  # vs sma_50
    assert "below" in text  # vs sma_200


def test_build_news_documents_handles_nan_event_types_without_printing_literal_nan():
    df = pd.DataFrame(
        {
            "symbol": ["ITC"],
            "news_id": ["xyz"],
            "headline": ["Some headline"],
            "published_timestamp": [pd.Timestamp("2024-01-01", tz="UTC")],
            "positive_probability": [0.5],
            "negative_probability": [0.5],
            "source": ["Reuters"],
            "event_types": [float("nan")],  # e.g. an empty string that round-tripped through CSV
        }
    )
    docs = build_news_documents(df)

    assert "nan" not in docs[0]["text"].lower()
    assert "none detected" in docs[0]["text"]


def test_build_news_documents_maps_provider_to_correct_tier_and_source():
    df = pd.DataFrame(
        {
            "symbol": ["ITC"],
            "news_id": ["abc123"],
            "headline": ["ITC declares dividend"],
            "published_timestamp": [pd.Timestamp("2024-01-01", tz="UTC")],
            "positive_probability": [0.8],
            "negative_probability": [0.1],
            "source": ["Reuters"],
            "event_types": ["dividend"],
        }
    )
    docs = build_news_documents(df)

    assert docs[0]["tier"] == TIER_MAJOR_NEWS
    assert docs[0]["source"] == "Reuters"
    assert "positive" in docs[0]["text"]


def test_build_risk_documents_uses_only_the_latest_row_per_symbol():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "symbol": ["A", "A"],
            "probability_up": [0.4, 0.6],
            "expected_volatility": [0.01, 0.02],
            "price_q10": [95.0, 96.0],
            "price_q50": [100.0, 101.0],
            "price_q90": [105.0, 106.0],
            "risk_label": ["Low", "Medium"],
            "risk_score": [20.0, 50.0],
        }
    )
    docs = build_risk_documents(df)

    assert len(docs) == 1
    assert docs[0]["tier"] == TIER_MODEL_OUTPUT
    assert "60.0%" in docs[0]["text"] or "60.%" in docs[0]["text"] or "60" in docs[0]["text"]
    assert "2024-01-02" in docs[0]["text"]


def test_build_explanation_documents_includes_top_factors():
    predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"]),
            "symbol": ["A"],
            "probability_up": [0.7],
            "confidence": [0.4],
            "confidence_label": ["Medium"],
        }
    )
    shap_values = pd.DataFrame({"feat_a": [0.5], "feat_b": [-0.3]}, index=predictions.index)

    def fake_top_factors(row, n=3):
        return {"positive": row[row > 0], "negative": row[row < 0]}

    docs = build_explanation_documents(predictions, shap_values, fake_top_factors, n_factors=3)

    assert "feat_a" in docs[0]["text"]
    assert "feat_b" in docs[0]["text"]
    assert docs[0]["tier"] == TIER_MODEL_OUTPUT
