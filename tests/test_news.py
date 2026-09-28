import pandas as pd
import pytest

from src.news.dedup import deduplicate_news, resolve_entity_mentions
from src.news.events import add_event_types, classify_event
from src.news.features import aggregate_daily_news_features
from src.news.impact import compute_event_occurrence_returns, explode_event_types, summarize_event_impact
from src.news.sentiment import add_sentiment, score_sentiment


# ---- dedup / entity resolution ----------------------------------------------


def test_deduplicate_news_drops_exact_duplicates():
    df = pd.DataFrame(
        {
            "news_id": ["a", "a", "b"],
            "symbol": ["RELIANCE", "RELIANCE", "RELIANCE"],
            "headline": ["x", "x", "y"],
        }
    )
    out = deduplicate_news(df)
    assert len(out) == 2


def test_resolve_entity_mentions_flags_genuine_references():
    df = pd.DataFrame(
        {
            "symbol": ["RELIANCE", "TCS"],
            "headline": ["Reliance Industries posts record profit", "Oil refiners in focus"],
            "summary": ["", "A broad sector roundup covering several unrelated energy companies"],
        }
    )
    out = resolve_entity_mentions(df)
    assert out["mentions_company"].tolist() == [True, False]


# ---- event classification ---------------------------------------------------


def test_classify_event_matches_known_keywords():
    assert "earnings" in classify_event("Company posts strong quarterly results")
    assert "dividend" in classify_event("Board declares special dividend")
    assert classify_event("A totally unrelated headline about weather") == []


def test_classify_event_can_match_multiple_types():
    matched = classify_event("Company acquires rival amid credit rating downgrade")
    assert "acquisition" in matched
    assert "credit_rating" in matched


def test_add_event_types_uses_headline_and_summary():
    df = pd.DataFrame({"headline": ["Routine update"], "summary": ["Board declares dividend"]})
    out = add_event_types(df)
    assert out["event_types"].iloc[0] == ["dividend"]


# ---- event impact -------------------------------------------------------------


def _daily_prices(symbol, closes, start="2024-01-01"):
    dates = pd.date_range(start, periods=len(closes), freq="D")
    return pd.DataFrame({"date": dates, "symbol": symbol, "close": closes})


def test_explode_event_types_creates_one_row_per_event_type():
    df = pd.DataFrame({"news_id": ["a", "b"], "event_types": [["earnings", "dividend"], []]})
    out = explode_event_types(df)
    assert out["event_type"].tolist() == ["earnings", "dividend"]


def test_compute_event_occurrence_returns_matches_hand_computed_forward_return():
    prices = _daily_prices("A", [100, 102, 104, 106, 108, 110])
    events = pd.DataFrame(
        {
            "event_type": ["earnings"],
            "symbol": ["A"],
            "published_timestamp": [pd.Timestamp("2024-01-01")],
        }
    )
    out = compute_event_occurrence_returns(events, prices, horizons=(1, 2))

    # base close on 2024-01-01 is 100; +1 day close=102 -> +2%; +2 days close=104 -> +4%
    assert out["return_1d"].iloc[0] == pytest.approx(0.02)
    assert out["return_2d"].iloc[0] == pytest.approx(0.04)


def test_compute_event_occurrence_returns_is_nan_beyond_available_history():
    prices = _daily_prices("A", [100, 102])
    events = pd.DataFrame(
        {"event_type": ["earnings"], "symbol": ["A"], "published_timestamp": [pd.Timestamp("2024-01-01")]}
    )
    out = compute_event_occurrence_returns(events, prices, horizons=(5,))
    assert pd.isna(out["return_5d"].iloc[0])


def test_summarize_event_impact_matches_hand_computed_mean():
    occurrences = pd.DataFrame(
        {
            "event_type": ["earnings", "earnings", "dividend"],
            "symbol": ["A", "B", "A"],
            "return_1d": [0.02, 0.04, -0.01],
        }
    )
    out = summarize_event_impact(occurrences, horizons=(1,)).set_index("event_type")

    assert out.loc["earnings", "occurrences"] == 2
    assert out.loc["earnings", "mean_return_1d"] == pytest.approx(0.03)
    assert out.loc["dividend", "occurrences"] == 1


# ---- sentiment (injected classify_fn, no real model load) -------------------


def test_score_sentiment_maps_labels_to_named_columns():
    def fake_classify(texts):
        return [
            [
                {"label": "positive", "score": 0.7},
                {"label": "neutral", "score": 0.2},
                {"label": "negative", "score": 0.1},
            ]
            for _ in texts
        ]

    out = score_sentiment(["good news"], fake_classify)

    assert out["positive_probability"].iloc[0] == pytest.approx(0.7)
    assert out["neutral_probability"].iloc[0] == pytest.approx(0.2)
    assert out["negative_probability"].iloc[0] == pytest.approx(0.1)


def test_aggregate_daily_news_features_matches_hand_computed_stats():
    df = pd.DataFrame(
        {
            "symbol": ["A", "A"],
            "published_timestamp": [pd.Timestamp("2024-01-01T09:00", tz="UTC")] * 2,
            "positive_probability": [0.8, 0.2],
            "negative_probability": [0.1, 0.6],
        }
    )
    out = aggregate_daily_news_features(df)
    row = out.iloc[0]

    assert row["news_count"] == 2
    assert row["sentiment_mean"] == pytest.approx(0.15)
    assert row["sentiment_std"] == pytest.approx(0.7778174593052023)
    assert row["positive_news_ratio"] == pytest.approx(0.5)
    assert row["negative_news_ratio"] == pytest.approx(0.5)


def test_add_sentiment_joins_scores_back_onto_the_news_dataframe():
    def fake_classify(texts):
        return [
            [{"label": "positive", "score": 1.0}, {"label": "neutral", "score": 0.0}, {"label": "negative", "score": 0.0}]
            for _ in texts
        ]

    df = pd.DataFrame({"headline": ["Great results"], "summary": ["Profit surges"]})
    out = add_sentiment(df, fake_classify)

    assert out["positive_probability"].iloc[0] == pytest.approx(1.0)
    assert out["headline"].iloc[0] == "Great results"
