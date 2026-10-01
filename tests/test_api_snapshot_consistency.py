"""Regression tests for the website audit's headline bug: different routes
independently picking different "latest date" per table, producing contradictory
numbers (e.g. a header close from a fresher table next to a forecast/entry-exit
computed off a staler one). Mirrors the real, independently-verified repo bug:
prices_daily/technical_features dated 2026-09-28 vs model_predictions/entry_exit
dated 2026-09-21, a week apart, because the notebooks producing them were re-run on
different days.
"""

import torch  # noqa: F401,E402

import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from src.api.main import app, get_db_engine
from src.db.schema import create_all_tables


def _make_client(prices_dates, technical_dates, prediction_dates, entry_exit_dates=()):
    engine = create_engine("sqlite:///:memory:", poolclass=StaticPool, connect_args={"check_same_thread": False})
    create_all_tables(engine)

    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(prices_dates),
            "symbol": ["ADANIENT"] * len(prices_dates),
            "open": [100.0] * len(prices_dates),
            "high": [102.0] * len(prices_dates),
            "low": [99.0] * len(prices_dates),
            "close": [2000.0 + i for i in range(len(prices_dates))],
            "adjusted_close": [2000.0 + i for i in range(len(prices_dates))],
            "volume": [1000.0] * len(prices_dates),
        }
    )
    prices.to_sql("prices_daily", engine, if_exists="append", index=False)

    technical_cols = {
        "date": pd.to_datetime(technical_dates),
        "symbol": ["ADANIENT"] * len(technical_dates),
        "open": [100.0] * len(technical_dates), "high": [102.0] * len(technical_dates),
        "low": [99.0] * len(technical_dates), "close": [2000.0 + i for i in range(len(technical_dates))],
        "volume": [1000.0] * len(technical_dates),
        "sma_20": [100.0] * len(technical_dates), "sma_50": [100.0] * len(technical_dates),
        "sma_200": [100.0] * len(technical_dates), "ema_20": [100.0] * len(technical_dates),
        "ema_50": [100.0] * len(technical_dates), "rsi_14": [50.0] * len(technical_dates),
        "macd": [0.1] * len(technical_dates), "macd_signal": [0.1] * len(technical_dates),
        "macd_histogram": [0.0] * len(technical_dates), "atr_14": [2.0] * len(technical_dates),
        "bb_upper": [105.0] * len(technical_dates), "bb_middle": [100.0] * len(technical_dates),
        "bb_lower": [95.0] * len(technical_dates), "bb_width": [0.1] * len(technical_dates),
        "volume_ratio": [1.0] * len(technical_dates), "technical_score": [50.0] * len(technical_dates),
    }
    pd.DataFrame(technical_cols).to_sql("technical_features", engine, if_exists="append", index=False)

    predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(prediction_dates),
            "symbol": ["ADANIENT"] * len(prediction_dates),
            "close": [2975.0 + i for i in range(len(prediction_dates))],
            "probability_up": [0.6] * len(prediction_dates),
            "expected_volatility": [0.02] * len(prediction_dates),
            "price_q10": [2926.88] * len(prediction_dates),
            "price_q50": [2975.0] * len(prediction_dates),
            "price_q90": [3023.41] * len(prediction_dates),
            "risk_score": [45.0] * len(prediction_dates),
            "risk_label": ["Medium"] * len(prediction_dates),
        }
    )
    predictions.to_sql("model_predictions", engine, if_exists="append", index=False)

    if entry_exit_dates:
        entry_exit = pd.DataFrame(
            {
                "date": pd.to_datetime(entry_exit_dates),
                "symbol": ["ADANIENT"] * len(entry_exit_dates),
                "has_signal": [True] * len(entry_exit_dates),
                "entry_low": [2975.0] * len(entry_exit_dates),
                "entry_high": [2990.0] * len(entry_exit_dates),
                "stop": [2859.25] * len(entry_exit_dates),
                "target": [3200.0] * len(entry_exit_dates),
            }
        )
        entry_exit.to_sql("entry_exit", engine, if_exists="append", index=False)

    app.dependency_overrides[get_db_engine] = lambda: engine
    test_client = TestClient(app)
    yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def misaligned_client():
    # prices_daily/technical_features run a week ahead of model_predictions/entry_exit --
    # exactly the confirmed real-repo gap (2026-09-28 vs 2026-09-21).
    fresh_dates = ["2026-09-19", "2026-09-21", "2026-09-28"]
    stale_dates = ["2026-09-19", "2026-09-21"]
    yield from _make_client(fresh_dates, fresh_dates, stale_dates, stale_dates)


@pytest.fixture
def aligned_client():
    same_dates = ["2026-09-19", "2026-09-21", "2026-09-28"]
    yield from _make_client(same_dates, same_dates, same_dates, same_dates)


def test_overview_and_forecast_report_the_same_snapshot_when_tables_diverge(misaligned_client):
    overview = misaligned_client.get("/stocks/ADANIENT/overview").json()
    forecast = misaligned_client.get("/stocks/ADANIENT/forecast").json()
    entry_exit = misaligned_client.get("/stocks/ADANIENT/entry-exit").json()

    # Both routes fall back to the oldest common date (2026-09-21), not each
    # independently reporting their own table's freshest date -- this is what
    # prevents the audit's reported header-vs-forecast contradiction.
    assert overview["snapshot_as_of"] == "2026-09-21"
    assert forecast["snapshot_as_of"] == "2026-09-21"
    assert entry_exit["snapshot_as_of"] == "2026-09-21"
    assert overview["snapshot_as_of"] == forecast["snapshot_as_of"] == entry_exit["snapshot_as_of"]

    # prices_daily/technical_features are flagged as lagging behind by 7 calendar
    # days (2026-09-28 vs 2026-09-21) instead of silently racing ahead unlabeled.
    assert overview["lagging_sources"]["technical_features"] == 7


def test_snapshot_route_reports_staleness_when_tables_diverge(misaligned_client):
    snapshot = misaligned_client.get("/stocks/ADANIENT/snapshot").json()

    assert snapshot["as_of"] == "2026-09-21"
    assert snapshot["is_fully_aligned"] is False
    assert snapshot["staleness_label"] == "very_stale"


def test_snapshot_as_of_equals_latest_date_when_all_tables_already_aligned(aligned_client):
    overview = aligned_client.get("/stocks/ADANIENT/overview").json()
    forecast = aligned_client.get("/stocks/ADANIENT/forecast").json()
    snapshot = aligned_client.get("/stocks/ADANIENT/snapshot").json()

    assert overview["snapshot_as_of"] == forecast["snapshot_as_of"] == "2026-09-28"
    assert snapshot["is_fully_aligned"] is True
    assert snapshot["lagging_sources"] == {}
    assert "lagging_sources" not in overview  # only attached when non-empty
