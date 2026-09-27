"""Downloads annual financial statements and the current valuation/ownership snapshot."""

from __future__ import annotations

import yfinance as yf

from src.ingestion.market_data import to_nse_ticker


def download_statements(symbol: str) -> dict:
    """Downloads annual income statement, balance sheet, and cash flow statement
    (fiscal-year columns, most-recent-first, as returned by Yahoo Finance),
    plus the current `.info` snapshot (valuation ratios, ownership %).
    """
    ticker = yf.Ticker(to_nse_ticker(symbol))
    return {
        "income_statement": ticker.financials,
        "balance_sheet": ticker.balance_sheet,
        "cash_flow": ticker.cashflow,
        "info": ticker.info,
    }
