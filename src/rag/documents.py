"""Builds the retrievable document corpus for the AI Analyst, from this
project's own already-computed pipeline outputs (Phases 2-10) -- not
fabricated or freely generated text.

Source reliability tiering (project plan Section 33). Tiering is applied
honestly based on *how this project actually obtained the data*, not on how
authoritative the underlying data ultimately is:

- Tier 1 (official): NSE/BSE/SEBI/company filings/RBI, fetched directly. This
  project does not currently scrape those directly (see Section 5's licensing
  notes) -- nothing is tagged Tier 1 yet.
- Tier 2 (major wire/publication): Reuters, Bloomberg, major financial press.
- Tier 3 (financial websites/analyst reports): this covers this project's
  Yahoo-Finance-sourced prices/fundamentals (relayed, not fetched directly from
  the exchange/regulator) and most news providers seen in Phase 6 (e.g. "Simply
  Wall St.").
- Tier 4 (blogs/forums/social): not currently used.
- "model_output": this project's own computed scores/predictions/explanations
  -- always labeled as an internal, not independently-verified, source.
"""

from __future__ import annotations

import pandas as pd

TIER_OFFICIAL = 1
TIER_MAJOR_NEWS = 2
TIER_FINANCIAL_WEBSITE = 3
TIER_SOCIAL = 4
TIER_MODEL_OUTPUT = "model_output"

TIER_LABELS = {
    TIER_OFFICIAL: "Tier 1 (official: NSE/BSE/SEBI/filings)",
    TIER_MAJOR_NEWS: "Tier 2 (major wire/publication)",
    TIER_FINANCIAL_WEBSITE: "Tier 3 (financial website/aggregator)",
    TIER_SOCIAL: "Tier 4 (blog/forum/social)",
    TIER_MODEL_OUTPUT: "Internal model output (not an independent source)",
}

PROVIDER_TIER_MAP = {
    "reuters": TIER_MAJOR_NEWS,
    "bloomberg": TIER_MAJOR_NEWS,
}


def provider_tier(provider_name: str) -> int:
    """Maps a news provider's display name to a source tier; defaults to Tier 3
    (financial website) for anything not recognized as a major wire service.
    """
    return PROVIDER_TIER_MAP.get(str(provider_name).strip().lower(), TIER_FINANCIAL_WEBSITE)


def _doc(doc_id, symbol, category, text, source, tier, date=None):
    return {
        "doc_id": doc_id,
        "symbol": symbol,
        "category": category,
        "text": text,
        "source": source,
        "tier": tier,
        "tier_label": TIER_LABELS[tier],
        "date": date,
    }


def build_fundamentals_documents(fundamentals_snapshot: pd.DataFrame) -> list[dict]:
    docs = []
    for symbol, row in fundamentals_snapshot.iterrows():
        text = (
            f"{symbol} latest fundamentals: ROE {row['roe']:.1%}, "
            f"Debt/Equity {row['debt_to_equity']:.2f}, revenue growth (YoY) {row['revenue_growth_yoy']:.1%}, "
            f"trailing P/E {row['pe_ratio']:.1f}. Fundamental Score: {row['fundamental_score']:.0f}/100."
        )
        docs.append(
            _doc(
                f"fund_{symbol}",
                symbol,
                "fundamentals",
                text,
                "Yahoo Finance (aggregated financial data)",
                TIER_FINANCIAL_WEBSITE,
            )
        )
    return docs


def build_technical_documents(technical_latest: pd.DataFrame) -> list[dict]:
    docs = []
    for _, row in technical_latest.iterrows():
        text = (
            f"{row['symbol']} technical snapshot on {row['date'].date()}: close Rs{row['close']:.2f}, "
            f"RSI(14) {row['rsi_14']:.1f}, MACD histogram {row['macd_histogram']:.2f}, "
            f"price vs SMA50 {'above' if row['close'] > row['sma_50'] else 'below'}, "
            f"price vs SMA200 {'above' if row['close'] > row['sma_200'] else 'below'}. "
            f"Technical Score: {row['technical_score']:.0f}/100."
        )
        docs.append(
            _doc(
                f"tech_{row['symbol']}_{row['date'].date()}",
                row["symbol"],
                "technical",
                text,
                "Yahoo Finance (relayed NSE EOD data)",
                TIER_FINANCIAL_WEBSITE,
                date=row["date"],
            )
        )
    return docs


def build_news_documents(news_scored: pd.DataFrame) -> list[dict]:
    docs = []
    for _, row in news_scored.iterrows():
        sentiment = "positive" if row["positive_probability"] > row["negative_probability"] else "negative"
        event_types = row["event_types"]
        # pd.isna() is required here, not `or`: an empty-string cell can round-trip
        # through CSV as NaN, and NaN is truthy in Python, so `event_types or
        # "none detected"` would silently print the literal text "nan" instead of
        # falling back.
        event_types_text = "none detected" if pd.isna(event_types) or not str(event_types) else str(event_types)
        text = (
            f"{row['symbol']} news ({row['published_timestamp']}): \"{row['headline']}\". "
            f"FinBERT sentiment: {sentiment} (positive={row['positive_probability']:.2f}, "
            f"negative={row['negative_probability']:.2f}). Event type(s): {event_types_text}."
        )
        docs.append(
            _doc(
                f"news_{row['symbol']}_{row['news_id']}",
                row["symbol"],
                "news",
                text,
                row["source"] or "Yahoo Finance News",
                provider_tier(row["source"]),
                date=pd.Timestamp(row["published_timestamp"]),
            )
        )
    return docs


def build_risk_documents(risk_output: pd.DataFrame) -> list[dict]:
    docs = []
    latest = risk_output.sort_values("date").groupby("symbol").tail(1)
    for _, row in latest.iterrows():
        text = (
            f"{row['symbol']} model outlook on {row['date'].date()}: probability of a next-day up move "
            f"{row['probability_up']:.1%}, expected daily volatility {row['expected_volatility']:.1%}, "
            f"predicted next-day range Rs{row['price_q10']:.1f} - Rs{row['price_q50']:.1f} - Rs{row['price_q90']:.1f}, "
            f"risk label: {row['risk_label']} ({row['risk_score']:.0f}/100)."
        )
        docs.append(
            _doc(
                f"risk_{row['symbol']}_{row['date'].date()}",
                row["symbol"],
                "model_prediction",
                text,
                "This project's own ML pipeline (Phase 8)",
                TIER_MODEL_OUTPUT,
                date=row["date"],
            )
        )
    return docs


def build_explanation_documents(
    explainability_predictions: pd.DataFrame, shap_values: pd.DataFrame, top_factors_fn, n_factors: int = 3
) -> list[dict]:
    docs = []
    latest = explainability_predictions.sort_values("date").groupby("symbol").tail(1)
    for idx, row in latest.iterrows():
        factors = top_factors_fn(shap_values.loc[idx], n=n_factors)
        positive_text = ", ".join(f"{k} (+{v:.3f})" for k, v in factors["positive"].items()) or "none"
        negative_text = ", ".join(f"{k} ({v:.3f})" for k, v in factors["negative"].items()) or "none"
        text = (
            f"{row['symbol']} prediction explanation on {row['date'].date()}: probability of up move "
            f"{row['probability_up']:.1%}, confidence {row['confidence']:.2f} ({row['confidence_label']}). "
            f"Top factors pushing bullish (log-odds units): {positive_text}. "
            f"Top factors pushing bearish (log-odds units): {negative_text}."
        )
        docs.append(
            _doc(
                f"explain_{row['symbol']}_{row['date'].date()}",
                row["symbol"],
                "explanation",
                text,
                "This project's own ML pipeline (Phase 10, SHAP)",
                TIER_MODEL_OUTPUT,
                date=row["date"],
            )
        )
    return docs
