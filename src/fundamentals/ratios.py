"""Per-fiscal-year fundamental ratios, growth metrics, and valuation/ownership snapshot.

Note: bank income statements (e.g. HDFCBANK) don't report EBIT/EBITDA or a
current-assets/current-liabilities split -- interest income/expense IS a bank's core
operating activity, not a below-the-line item. roce, current_ratio, and
interest_coverage are therefore NaN for banks; roe, debt_to_equity, and the growth
metrics are unaffected and still compute normally.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

GROWTH_COLUMNS = ("revenue", "ebitda", "eps_diluted", "free_cash_flow")


def _get_row(statement: pd.DataFrame, label: str) -> pd.Series:
    """Returns a fiscal-year-indexed row from a statement, or all-NaN if the label is absent."""
    if label in statement.index:
        return statement.loc[label]
    return pd.Series(np.nan, index=statement.columns)


def build_annual_fundamentals(statements: dict) -> pd.DataFrame:
    """Combines income statement, balance sheet, and cash flow into one per-fiscal-year
    table (oldest fiscal year first), with the core ratios computed.
    """
    income = statements["income_statement"]
    balance = statements["balance_sheet"]
    cash = statements["cash_flow"]

    data = pd.DataFrame(index=income.columns)
    data["revenue"] = _get_row(income, "Total Revenue")
    data["ebitda"] = _get_row(income, "EBITDA")
    data["ebit"] = _get_row(income, "EBIT")
    data["net_income"] = _get_row(income, "Net Income")
    data["eps_diluted"] = _get_row(income, "Diluted EPS")
    data["interest_expense"] = _get_row(income, "Interest Expense")

    data["total_assets"] = _get_row(balance, "Total Assets")
    data["stockholders_equity"] = _get_row(balance, "Stockholders Equity")
    data["total_debt"] = _get_row(balance, "Total Debt")
    data["current_assets"] = _get_row(balance, "Current Assets")
    data["current_liabilities"] = _get_row(balance, "Current Liabilities")
    data["invested_capital"] = _get_row(balance, "Invested Capital")

    data["operating_cash_flow"] = _get_row(cash, "Operating Cash Flow")
    data["capital_expenditure"] = _get_row(cash, "Capital Expenditure")
    data["free_cash_flow"] = _get_row(cash, "Free Cash Flow")

    data["roe"] = data["net_income"] / data["stockholders_equity"]
    data["roce"] = data["ebit"] / data["invested_capital"]
    data["debt_to_equity"] = data["total_debt"] / data["stockholders_equity"]
    data["current_ratio"] = data["current_assets"] / data["current_liabilities"]
    data["interest_coverage"] = data["ebit"] / data["interest_expense"].replace(0, np.nan)

    return data.sort_index()


def add_growth_metrics(annual: pd.DataFrame) -> pd.DataFrame:
    """Adds {col}_growth_yoy = pct_change() for each column in GROWTH_COLUMNS.

    `annual` must be sorted oldest-to-newest fiscal year (as returned by
    build_annual_fundamentals) so pct_change() compares each year to the prior one.
    """
    annual = annual.copy()
    for col in GROWTH_COLUMNS:
        annual[f"{col}_growth_yoy"] = annual[col].pct_change()
    return annual


def cagr(series: pd.Series) -> float:
    """Compound annual growth rate from the first to the last non-NaN value.

    `series` must be sorted oldest-to-newest.
    """
    clean = series.dropna()
    if len(clean) < 2 or clean.iloc[0] <= 0:
        return float("nan")
    years = len(clean) - 1
    return (clean.iloc[-1] / clean.iloc[0]) ** (1 / years) - 1


def snapshot_valuation_and_ownership(info: dict) -> dict:
    """Current-day valuation ratios and ownership %, read from the yfinance .info snapshot.

    Note: `institutional_holding` is Yahoo's aggregate institutional-holding figure,
    not split into FII/DII as NSE/BSE disclosures do; `promoter_holding_proxy` uses
    Yahoo's insider-holding figure as an approximation of NSE's promoter-holding
    disclosure. Pledged-share data is not available from this source.
    """
    return {
        "pe_ratio": info.get("trailingPE"),
        "pb_ratio": info.get("priceToBook"),
        "ps_ratio": info.get("priceToSalesTrailing12Months"),
        "ev_to_ebitda": info.get("enterpriseToEbitda"),
        "dividend_yield": info.get("dividendYield"),
        "promoter_holding_proxy": info.get("heldPercentInsiders"),
        "institutional_holding": info.get("heldPercentInstitutions"),
    }
