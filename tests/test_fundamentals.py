import numpy as np
import pandas as pd
import pytest

from src.fundamentals.ratios import (
    add_growth_metrics,
    build_annual_fundamentals,
    cagr,
    snapshot_valuation_and_ownership,
)
from src.fundamentals.score import fundamental_score


def _statements():
    # Two fiscal years, oldest last (as yfinance returns: most-recent-first).
    fy2024 = pd.Timestamp("2024-03-31")
    fy2023 = pd.Timestamp("2023-03-31")
    columns = [fy2024, fy2023]

    income = pd.DataFrame(
        {
            fy2024: [1000.0, 200.0, 150.0, 100.0, 10.0, 20.0],
            fy2023: [800.0, 150.0, 100.0, 70.0, 7.0, 15.0],
        },
        index=["Total Revenue", "EBITDA", "EBIT", "Net Income", "Diluted EPS", "Interest Expense"],
    )[columns]

    balance = pd.DataFrame(
        {
            fy2024: [2000.0, 500.0, 200.0, 300.0, 150.0, 600.0],
            fy2023: [1800.0, 450.0, 200.0, 250.0, 130.0, 550.0],
        },
        index=[
            "Total Assets",
            "Stockholders Equity",
            "Total Debt",
            "Current Assets",
            "Current Liabilities",
            "Invested Capital",
        ],
    )[columns]

    cash = pd.DataFrame(
        {fy2024: [120.0, -30.0, 90.0], fy2023: [100.0, -25.0, 75.0]},
        index=["Operating Cash Flow", "Capital Expenditure", "Free Cash Flow"],
    )[columns]

    return {"income_statement": income, "balance_sheet": balance, "cash_flow": cash}


def test_build_annual_fundamentals_matches_hand_computed_ratios():
    annual = build_annual_fundamentals(_statements())

    fy2023 = pd.Timestamp("2023-03-31")
    fy2024 = pd.Timestamp("2024-03-31")
    assert list(annual.index) == [fy2023, fy2024]  # sorted oldest-first

    # FY2024 balance sheet: total_assets=2000, equity=500, debt=200,
    # current_assets=300, current_liabilities=150, invested_capital=600.
    row = annual.loc[fy2024]
    assert row["roe"] == pytest.approx(100 / 500)
    assert row["roce"] == pytest.approx(150 / 600)
    assert row["debt_to_equity"] == pytest.approx(200 / 500)
    assert row["current_ratio"] == pytest.approx(300 / 150)
    assert row["interest_coverage"] == pytest.approx(150 / 20)


def test_build_annual_fundamentals_handles_missing_fields_as_nan():
    statements = _statements()
    statements["income_statement"] = statements["income_statement"].drop(index="EBIT")

    annual = build_annual_fundamentals(statements)

    assert annual["ebit"].isna().all()
    assert annual["roce"].isna().all()  # depends on ebit
    assert not annual["roe"].isna().any()  # unaffected


def test_interest_coverage_is_nan_not_inf_when_interest_expense_is_zero():
    statements = _statements()
    statements["income_statement"].loc["Interest Expense"] = [0.0, 0.0]

    annual = build_annual_fundamentals(statements)

    assert annual["interest_coverage"].isna().all()


def test_add_growth_metrics_matches_hand_computed_yoy_change():
    annual = build_annual_fundamentals(_statements())
    out = add_growth_metrics(annual)

    fy2024 = pd.Timestamp("2024-03-31")
    assert out.loc[fy2024, "revenue_growth_yoy"] == pytest.approx((1000 - 800) / 800)
    assert out.loc[fy2024, "eps_diluted_growth_yoy"] == pytest.approx((10 - 7) / 7)


def test_cagr_matches_hand_computed_compound_growth():
    # 100 -> 110 -> 121 over 2 years is an exact 10% CAGR (121 = 100 * 1.1^2)
    series = pd.Series([100.0, 110.0, 121.0])
    assert cagr(series) == pytest.approx(0.10)


def test_cagr_is_nan_for_a_single_data_point():
    assert np.isnan(cagr(pd.Series([100.0])))


def test_snapshot_valuation_and_ownership_reads_expected_info_keys():
    info = {
        "trailingPE": 22.5,
        "priceToBook": 3.1,
        "heldPercentInsiders": 0.55,
        "heldPercentInstitutions": 0.2,
    }
    out = snapshot_valuation_and_ownership(info)

    assert out["pe_ratio"] == 22.5
    assert out["pb_ratio"] == 3.1
    assert out["promoter_holding_proxy"] == 0.55
    assert out["institutional_holding"] == 0.2
    assert out["dividend_yield"] is None  # not present in `info` -> .get() default


def test_fundamental_score_matches_hand_computed_average_of_subscores():
    df = pd.DataFrame(
        {
            "roe": [0.20],  # -> 50
            "debt_to_equity": [1.0],  # -> 50
            "revenue_growth_yoy": [0.10],  # -> (0.10+0.20)/0.60*100 = 50
            "pe_ratio": [20.0],  # -> (1-20/40)*100 = 50
        }
    )
    out = fundamental_score(df)

    assert out.iloc[0] == pytest.approx(50.0)


def test_fundamental_score_is_nan_when_a_required_input_is_missing():
    df = pd.DataFrame(
        {
            "roe": [0.20],
            "debt_to_equity": [1.0],
            "revenue_growth_yoy": [0.10],
            "pe_ratio": [float("nan")],
        }
    )
    out = fundamental_score(df)

    assert pd.isna(out.iloc[0])
