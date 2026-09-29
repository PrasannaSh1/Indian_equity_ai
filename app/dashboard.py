"""Streamlit investor research dashboard (project plan Section 34, Phase 13).

Reads everything from the FastAPI backend (src/api/main.py) -- this file is
pure presentation, no model logic or database access, so it stays simple and
the API client (app/api_client.py) can be unit-tested independently.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import requests
import streamlit as st

from app.api_client import ApiClient
from app.charts import OSCILLATOR_OPTIONS, OVERLAY_OPTIONS, build_price_chart

st.set_page_config(page_title="Indian Equity AI Analyst", layout="wide")

DISCLAIMER = (
    "This is a research/decision-support tool. Outputs are model-generated estimates, "
    "not guaranteed predictions or investment advice."
)


@st.cache_resource
def get_client() -> ApiClient:
    return ApiClient()


def safe_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs), None
    except requests.RequestException as exc:
        return None, str(exc)


client = get_client()

st.title("Indian Equity AI Analyst")
st.caption(DISCLAIMER)

stocks, error = safe_call(client.list_stocks)
if error:
    st.error(f"Could not reach the API backend at {client.base_url}: {error}")
    st.info("Start it with: uvicorn src.api.main:app --reload")
    st.stop()

symbol = st.selectbox("Search a stock", stocks)

overview, _ = safe_call(client.overview, symbol)
forecast, _ = safe_call(client.forecast, symbol)

if overview:
    col1, col2, col3 = st.columns(3)
    col1.metric(f"{symbol} — Close", f"Rs{overview['close']:.2f}")
    if forecast:
        direction = "Bullish" if forecast["probability_up"] > 0.5 else "Bearish"
        col2.metric("AI Outlook", direction, f"{forecast['probability_up']:.1%} P(up)")
        col3.metric("Risk", forecast["risk_label"], f"{forecast['risk_score']:.0f}/100")

tabs = st.tabs(
    ["Overview", "Technical", "Fundamentals", "News", "Sentiment", "AI Forecast", "Risk", "Backtest", "AI Analyst"]
)

with tabs[0]:
    st.subheader("Overview")
    st.caption(
        "Daily EOD price chart (this project's data is end-of-day, not streaming intraday ticks) "
        "with optional indicator overlays."
    )

    chart_cols = st.columns(2)
    selected_overlays = chart_cols[0].multiselect("Overlay indicators", OVERLAY_OPTIONS, default=["SMA 50"])
    selected_oscillators = chart_cols[1].multiselect("Oscillator panels", OSCILLATOR_OPTIONS)
    lookback_days = st.slider("Lookback (trading days)", min_value=30, max_value=500, value=180, step=10)

    history, hist_err = safe_call(client.history, symbol, days=lookback_days)
    if history:
        history_df = pd.DataFrame(history)
        history_df["date"] = pd.to_datetime(history_df["date"])
        fig = build_price_chart(history_df, overlays=selected_overlays, oscillators=selected_oscillators)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(hist_err or "No price history available for this symbol.")

    if overview:
        with st.expander("Raw overview data"):
            st.json(overview)
    else:
        st.warning("No overview data available for this symbol.")

with tabs[1]:
    st.subheader("Technical Analysis")
    technical, err = safe_call(client.technical, symbol)
    if technical:
        st.metric("Technical Score", f"{technical['technical_score']:.0f}/100")
        cols = st.columns(3)
        cols[0].metric("RSI(14)", f"{technical['rsi_14']:.1f}")
        cols[1].metric("MACD Histogram", f"{technical['macd_histogram']:.2f}")
        cols[2].metric("ATR(14)", f"{technical['atr_14']:.2f}")
        st.write(
            f"SMA20: Rs{technical['sma_20']:.2f} | SMA50: Rs{technical['sma_50']:.2f} | "
            f"SMA200: Rs{technical['sma_200']:.2f}"
        )
    else:
        st.warning(err or "No technical data available.")

with tabs[2]:
    st.subheader("Fundamentals")
    fundamentals, err = safe_call(client.fundamentals, symbol)
    if fundamentals:
        st.metric("Fundamental Score", f"{fundamentals['fundamental_score']:.0f}/100")
        cols = st.columns(4)
        cols[0].metric("ROE", f"{fundamentals['roe']:.1%}")
        cols[1].metric("Debt/Equity", f"{fundamentals['debt_to_equity']:.2f}")
        cols[2].metric("Revenue Growth (YoY)", f"{fundamentals['revenue_growth_yoy']:.1%}")
        cols[3].metric("P/E", f"{fundamentals['pe_ratio']:.1f}")
    else:
        st.warning(err or "No fundamentals data available.")

with tabs[3]:
    st.subheader("News")
    news, err = safe_call(client.news, symbol)
    if news:
        for item in news:
            st.markdown(f"**{item['headline']}**  \n*{item['source']} — {item['published_timestamp']}*")
    else:
        st.info(err or "No recent news available for this symbol.")

with tabs[4]:
    st.subheader("Sentiment")
    news, err = safe_call(client.news, symbol)
    if news:
        df = pd.DataFrame(news)
        avg_positive = df["positive_probability"].mean()
        avg_negative = df["negative_probability"].mean()
        cols = st.columns(2)
        cols[0].metric("Avg. Positive Sentiment", f"{avg_positive:.1%}")
        cols[1].metric("Avg. Negative Sentiment", f"{avg_negative:.1%}")
        st.bar_chart(df.set_index("headline")[["positive_probability", "negative_probability"]])
    else:
        st.info(err or "No sentiment data available for this symbol.")

with tabs[5]:
    st.subheader("AI Forecast")
    if forecast:
        st.write(f"Probability of next-day up move: **{forecast['probability_up']:.1%}**")
        st.write(f"Expected daily volatility: **{forecast['expected_volatility']:.1%}**")
        st.write(
            f"Predicted next-day range: Rs{forecast['price_q10']:.2f} — "
            f"Rs{forecast['price_q50']:.2f} — Rs{forecast['price_q90']:.2f}"
        )
        entry_exit, _ = safe_call(client.entry_exit, symbol)
        if entry_exit and entry_exit.get("has_signal"):
            st.markdown("**Entry/Exit (hypothetical, not guaranteed):**")
            st.write(f"Entry Zone: Rs{entry_exit['entry_low']:.2f} — Rs{entry_exit['entry_high']:.2f}")
            st.write(f"Stop Zone: Rs{entry_exit['stop']:.2f}")
            st.write(f"Target Zone: Rs{entry_exit['target']:.2f}")
    else:
        st.warning("No forecast data available.")

with tabs[6]:
    st.subheader("Risk")
    risk, err = safe_call(client.risk, symbol)
    explanation, _ = safe_call(client.explanation, symbol)
    if risk:
        st.metric("Risk Label", risk["risk_label"], f"{risk['risk_score']:.0f}/100")
    if explanation:
        st.write(f"Model confidence: **{explanation['confidence_label']}** ({explanation['confidence']:.2f})")
    if not risk and not explanation:
        st.warning(err or "No risk data available.")

with tabs[7]:
    st.subheader("Backtest")
    backtest, err = safe_call(client.backtest)
    if backtest:
        df = pd.DataFrame(backtest).pivot(index="metric", columns="system", values="value")
        st.dataframe(df)
    else:
        st.warning(err or "No backtest results available.")

with tabs[8]:
    st.subheader("AI Analyst")
    st.caption("Answers are composed only from retrieved, cited evidence -- see each source below.")
    query = st.text_input("Ask a question about this stock", placeholder="Why is this stock bullish?")
    if st.button("Ask") and query:
        result, err = safe_call(client.ask, symbol, query)
        if result:
            st.write(result["answer"])
        else:
            st.error(err or "Could not get an answer.")
