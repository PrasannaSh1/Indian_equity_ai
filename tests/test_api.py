# torch must be imported before pandas/pyarrow anywhere in this process (see
# src/news/sentiment.py) -- src.api.main imports torch first internally, but
# that only helps if nothing earlier in *this* process (e.g. this test file's
# own imports) has already pulled in pandas/pyarrow first.
import torch  # noqa: F401,E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

import src.api.main as api_main
from src.api.main import app, get_db_engine, get_rag_index
from src.db.schema import create_all_tables
from src.rag.retrieval import VectorIndex


@pytest.fixture
def client():
    # StaticPool is required: FastAPI runs sync route handlers in a threadpool, and
    # plain sqlite:///:memory: gives each new connection its own empty database --
    # StaticPool makes every connection share the same underlying in-memory DB.
    engine = create_engine(
        "sqlite:///:memory:", poolclass=StaticPool, connect_args={"check_same_thread": False}
    )
    create_all_tables(engine)

    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "symbol": ["RELIANCE", "RELIANCE"],
            "open": [100.0, 101.0], "high": [102.0, 103.0], "low": [99.0, 100.0],
            "close": [101.0, 102.0], "adjusted_close": [101.0, 102.0], "volume": [1000.0, 1100.0],
        }
    )
    prices.to_sql("prices_daily", engine, if_exists="append", index=False)

    technical = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "symbol": ["RELIANCE"] * 3,
            "open": [100.0, 101.0, 102.0], "high": [102.0, 103.0, 104.0], "low": [99.0, 100.0, 101.0],
            "close": [101.0, 102.0, 103.0], "volume": [1000.0, 1100.0, 1200.0],
            "sma_20": [100.0, 101.0, 102.0], "sma_50": [None, None, None], "sma_200": [None, None, None],
            "ema_20": [100.0, 101.0, 102.0], "ema_50": [100.0, 101.0, 102.0],
            "rsi_14": [50.0, 55.0, 60.0], "macd": [0.1, 0.2, 0.3], "macd_signal": [0.1, 0.1, 0.2],
            "macd_histogram": [0.0, 0.1, 0.1], "atr_14": [2.0, 2.1, 2.2],
            "bb_upper": [105.0, 106.0, 107.0], "bb_middle": [100.0, 101.0, 102.0], "bb_lower": [95.0, 96.0, 97.0],
            "bb_width": [0.1, 0.1, 0.1], "volume_ratio": [1.0, 1.1, 1.2], "technical_score": [50.0, 55.0, 60.0],
        }
    )
    technical.to_sql("technical_features", engine, if_exists="append", index=False)

    predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"]), "symbol": ["RELIANCE"], "close": [102.0],
            "probability_up": [0.6], "expected_volatility": [0.02], "price_q10": [98.0],
            "price_q50": [102.0], "price_q90": [106.0], "risk_score": [45.0], "risk_label": ["Medium"],
        }
    )
    predictions.to_sql("model_predictions", engine, if_exists="append", index=False)

    fundamentals = pd.DataFrame(
        {
            "symbol": ["RELIANCE"], "roe": [0.09], "roce": [0.12], "debt_to_equity": [0.4],
            "current_ratio": [1.2], "revenue_growth_yoy": [0.1], "pe_ratio": [22.0],
            "pb_ratio": [1.8], "dividend_yield": [0.005], "fundamental_score": [50.0],
        }
    )
    fundamentals.to_sql("fundamentals_snapshot", engine, if_exists="append", index=False)

    # win_rate/profit_factor are legitimately NaN for a pure buy-and-hold system --
    # this must round-trip as JSON null, not crash the response (see test below).
    backtest = pd.DataFrame(
        {
            "system": ["strategy", "nifty50_buy_and_hold"],
            "metric": ["win_rate", "win_rate"],
            "value": [0.5, float("nan")],
        }
    )
    backtest.to_sql("backtest_results", engine, if_exists="append", index=False)

    def fake_embed(texts):
        return np.ones((len(texts), 2))

    rag_index = VectorIndex(fake_embed)
    rag_index.build(
        [
            {
                "doc_id": "d1", "symbol": "RELIANCE", "category": "test", "text": "Reliance is doing fine",
                "source": "Test", "tier": 3, "tier_label": "Tier 3 (financial website/aggregator)",
            }
        ]
    )

    app.dependency_overrides[get_db_engine] = lambda: engine
    app.dependency_overrides[get_rag_index] = lambda: rag_index
    test_client = TestClient(app)
    yield test_client
    app.dependency_overrides.clear()


def test_health():
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}


def test_list_stocks_returns_distinct_symbols(client):
    resp = client.get("/stocks")
    assert resp.status_code == 200
    assert resp.json() == ["RELIANCE"]


def test_overview_returns_the_latest_row(client):
    resp = client.get("/stocks/RELIANCE/overview")
    assert resp.status_code == 200
    assert resp.json()["close"] == pytest.approx(102.0)  # the later of the two dates


def test_overview_404s_for_an_unknown_symbol(client):
    resp = client.get("/stocks/UNKNOWN/overview")
    assert resp.status_code == 404


def test_forecast_and_risk_endpoints(client):
    forecast = client.get("/stocks/RELIANCE/forecast").json()
    assert forecast["probability_up"] == pytest.approx(0.6)
    assert forecast["price_q50"] == pytest.approx(102.0)

    risk = client.get("/stocks/RELIANCE/risk").json()
    assert risk["risk_score"] == 45.0
    assert risk["risk_label"] == "Medium"
    # model_predictions (this route's source table) is dated 2024-01-02 while the
    # fixture's technical_features runs one day ahead (2024-01-03) -- risk correctly
    # reports the shared, older snapshot date rather than silently mixing in the
    # newer one, and surfaces the lag instead of hiding it.
    assert risk["snapshot_as_of"] == "2024-01-02"
    assert risk["lagging_sources"] == {"technical_features": 1}


def test_history_endpoint_returns_ascending_dates_and_respects_days_limit(client):
    resp = client.get("/stocks/RELIANCE/history", params={"days": 2})
    assert resp.status_code == 200
    rows = resp.json()

    assert len(rows) == 2  # limited to the 2 most recent rows...
    dates = [r["date"] for r in rows]
    assert dates == sorted(dates)  # ...but returned oldest-to-newest for charting
    assert rows[-1]["close"] == pytest.approx(103.0)  # the most recent row


def test_history_endpoint_serializes_nan_sma_as_null(client):
    resp = client.get("/stocks/RELIANCE/history", params={"days": 10})
    rows = resp.json()
    assert rows[0]["sma_50"] is None


def test_fundamentals_endpoint(client):
    resp = client.get("/stocks/RELIANCE/fundamentals").json()
    assert resp["fundamental_score"] == pytest.approx(50.0)


def test_backtest_endpoint_serializes_nan_as_json_null_not_a_500(client):
    resp = client.get("/backtest")
    assert resp.status_code == 200
    rows = resp.json()
    nifty_row = next(r for r in rows if r["system"] == "nifty50_buy_and_hold")
    assert nifty_row["value"] is None
    strategy_row = next(r for r in rows if r["system"] == "strategy")
    assert strategy_row["value"] == pytest.approx(0.5)


def test_ai_analyst_ask_returns_a_grounded_cited_answer(client):
    resp = client.post("/ai-analyst/ask", json={"symbol": "RELIANCE", "query": "why is this bullish"})
    assert resp.status_code == 200
    body = resp.json()
    assert "Reliance is doing fine" in body["answer"]
    assert body["sources"][0]["doc_id"] == "d1"


def test_resolve_company_endpoint_known_ticker():
    client = TestClient(app)
    resp = client.get("/company/TCS/resolve")
    assert resp.status_code == 200
    body = resp.json()
    assert body["symbol"] == "TCS"
    assert body["in_training_universe"] is True


def test_resolve_company_endpoint_unseen_ticker():
    client = TestClient(app)
    resp = client.get("/company/DIXON/resolve")
    assert resp.status_code == 200
    assert resp.json()["in_training_universe"] is False


def test_resolve_company_endpoint_422s_for_empty_identifier():
    client = TestClient(app)
    resp = client.get("/company/%20/resolve")
    assert resp.status_code == 422


def test_analysis_endpoint_runs_the_live_pipeline(monkeypatch):
    """Phase 26: the orchestration internals are covered by
    tests/services/test_company_analysis.py -- this only checks that the FastAPI
    route wires the request through to analyze_company() and returns its result.
    """
    fake_result = {"company": {"symbol": "TCS"}, "model_coverage": {"in_training_universe": True}}
    monkeypatch.setattr(
        api_main,
        "analyze_company",
        lambda company, horizon="5d", analysis_type="full": fake_result,
    )

    client = TestClient(app)
    resp = client.post("/analysis", json={"company": "TCS", "horizon": "5d", "analysis_type": "quick"})

    assert resp.status_code == 200
    assert resp.json() == fake_result


def test_lineage_endpoint_returns_one_row_per_category(client):
    resp = client.get("/stocks/RELIANCE/lineage")
    assert resp.status_code == 200
    categories = {row["category"] for row in resp.json()}
    assert categories == {
        "market_data",
        "fundamentals_balance_sheet",
        "fundamentals_valuation",
        "news",
        "model_classifier",
        "model_quantile",
    }


def test_lineage_endpoint_reports_unavailable_fundamentals_period_when_not_recorded(client):
    # The fixture's fundamentals_snapshot row predates the fiscal_year/valuation_as_of
    # columns -- this must degrade to "unavailable", not crash or fabricate a date.
    resp = client.get("/stocks/RELIANCE/lineage")
    balance_sheet_row = next(r for r in resp.json() if r["category"] == "fundamentals_balance_sheet")
    assert balance_sheet_row["as_of_or_period"] == "unavailable"


def test_news_endpoint_falls_back_to_live_fetch_when_precomputed_is_empty(client, monkeypatch):
    live_articles = pd.DataFrame(
        {
            "news_id": ["id1"],
            "symbol": ["RELIANCE"],
            "headline": ["Live headline"],
            "summary": [""],
            "source": ["Yahoo"],
            "url": [""],
            "published_timestamp": [pd.Timestamp("2026-09-30", tz="UTC")],
        }
    )
    monkeypatch.setattr(
        api_main, "fetch_with_fallback", lambda symbol: (live_articles, {"provider": "yahoo_finance", "retrieved_at": "2026-09-30T00:00:00+00:00"})
    )
    monkeypatch.setattr(
        api_main,
        "add_sentiment",
        lambda articles, classify_fn: articles.assign(positive_probability=0.7, negative_probability=0.1, neutral_probability=0.2),
    )
    monkeypatch.setattr(api_main, "_default_sentiment_classify_fn", lambda: None)

    resp = client.get("/stocks/RELIANCE/news")

    assert resp.status_code == 200
    records = resp.json()
    assert len(records) == 1
    assert records[0]["source_mode"] == "live_fallback"
    assert records[0]["retrieved_at"] == "2026-09-30T00:00:00+00:00"
    assert records[0]["headline"] == "Live headline"


def test_analysis_endpoint_returns_422_for_unresolvable_company(monkeypatch):
    def raise_not_resolved(company, horizon="5d", analysis_type="full"):
        from src.security.resolver import CompanyNotResolvedError

        raise CompanyNotResolvedError("nope")

    monkeypatch.setattr(api_main, "analyze_company", raise_not_resolved)

    client = TestClient(app)
    resp = client.post("/analysis", json={"company": "###"})

    assert resp.status_code == 422
