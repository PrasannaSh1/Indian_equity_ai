"""Builds the interactive price/indicator chart for the dashboard.

A pure function with no Streamlit dependency, so it's unit-testable directly
by inspecting the returned figure's traces.

Note: this project only has daily EOD data (Phase 1), not streaming intraday
ticks (see PROJECT_PLAN.md Phase 12's intraday note) -- "real-time" here means
an interactive, up-to-date daily chart, not live tick-by-tick data.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

OVERLAY_OPTIONS = ["SMA 20", "SMA 50", "SMA 200", "EMA 20", "EMA 50", "Bollinger Bands"]
OSCILLATOR_OPTIONS = ["RSI", "MACD"]

_OVERLAY_COLUMNS = {
    "SMA 20": [("sma_20", "SMA 20")],
    "SMA 50": [("sma_50", "SMA 50")],
    "SMA 200": [("sma_200", "SMA 200")],
    "EMA 20": [("ema_20", "EMA 20")],
    "EMA 50": [("ema_50", "EMA 50")],
    "Bollinger Bands": [
        ("bb_upper", "BB Upper"),
        ("bb_middle", "BB Middle"),
        ("bb_lower", "BB Lower"),
    ],
}


def build_price_chart(
    history: pd.DataFrame, overlays: list[str] | None = None, oscillators: list[str] | None = None
) -> go.Figure:
    """`history` must have columns: date, open, high, low, close, volume, plus
    whichever indicator columns are needed for the requested overlays/oscillators
    (missing columns are silently skipped, not an error, since not every symbol's
    history is guaranteed to have every column populated this early in warm-up).
    """
    overlays = overlays or []
    oscillators = oscillators or []
    n_oscillators = len(oscillators)
    total_rows = 2 + n_oscillators

    if n_oscillators:
        row_heights = [0.55, 0.15] + [0.30 / n_oscillators] * n_oscillators
    else:
        row_heights = [0.7, 0.3]

    fig = make_subplots(
        rows=total_rows, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=row_heights
    )

    fig.add_trace(
        go.Candlestick(
            x=history["date"], open=history["open"], high=history["high"],
            low=history["low"], close=history["close"], name="Price",
        ),
        row=1, col=1,
    )

    for overlay in overlays:
        for column, label in _OVERLAY_COLUMNS.get(overlay, []):
            if column in history.columns:
                fig.add_trace(
                    go.Scatter(x=history["date"], y=history[column], mode="lines", name=label, line=dict(width=1)),
                    row=1, col=1,
                )

    fig.add_trace(go.Bar(x=history["date"], y=history["volume"], name="Volume"), row=2, col=1)

    for row, oscillator in enumerate(oscillators, start=3):
        if oscillator == "RSI" and "rsi_14" in history.columns:
            fig.add_trace(go.Scatter(x=history["date"], y=history["rsi_14"], name="RSI(14)"), row=row, col=1)
        elif oscillator == "MACD" and "macd" in history.columns:
            fig.add_trace(go.Scatter(x=history["date"], y=history["macd"], name="MACD"), row=row, col=1)
            fig.add_trace(
                go.Scatter(x=history["date"], y=history["macd_signal"], name="MACD Signal"), row=row, col=1
            )
            fig.add_trace(
                go.Bar(x=history["date"], y=history["macd_histogram"], name="MACD Histogram"), row=row, col=1
            )

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=450 + 150 * n_oscillators,
        showlegend=True,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig
