"""FastAPI backend serving predictions/risk/explanations (project plan Phase 13).

Reads from the SQLite app database (src/db/schema.py) populated by
src/db/load_data.py from Phases 1-12's already-computed outputs -- this app
serves precomputed research results, it does not retrain models per request.
"""

# IMPORTANT (Windows): torch must be imported before pandas/pyarrow anywhere in
# this process, or its native DLL load can fail -- see src/news/sentiment.py.
# The AI Analyst endpoint below depends on sentence-transformers (-> torch),
# and everything else in this app imports pandas, so this must stay first.
import torch  # noqa: F401,E402

import sys  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402
from datetime import date  # noqa: E402
from pathlib import Path  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
from fastapi import Depends, FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from src.db.schema import get_engine  # noqa: E402
from src.explainability.shap_explainer import top_factors  # noqa: E402
from src.models.registry import QUANTILE_REGISTRY_FILENAME, load_latest  # noqa: E402
from src.news.providers.chain import fetch_with_fallback  # noqa: E402
from src.news.sentiment import add_sentiment  # noqa: E402
from src.rag.analyst import answer_query  # noqa: E402
from src.rag.documents import (  # noqa: E402
    TIER_FINANCIAL_WEBSITE,
    TIER_LABELS,
    TIER_MODEL_OUTPUT,
    build_explanation_documents,
    build_fundamentals_documents,
    build_news_documents,
    build_risk_documents,
    build_technical_documents,
    provider_tier,
)
from src.rag.retrieval import VectorIndex, load_sentence_transformer_embedder  # noqa: E402
from src.security.resolver import (  # noqa: E402
    AmbiguousCompanyError,
    CompanyNotResolvedError,
    resolve,
)
from src.services.company_analysis import _default_sentiment_classify_fn, analyze_company  # noqa: E402
from src.validation.freshness import staleness_label, trading_days_stale  # noqa: E402
from src.validation.snapshot import resolve_snapshot  # noqa: E402

DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

# Tables that each independently track a "latest date" for a symbol. Root cause of
# the header/forecast/entry-exit contradiction bug: these can silently drift apart
# (e.g. prices_daily refreshed more recently than model_predictions, because the
# notebooks that produce them were last re-run on different days) if each route
# picks its own table's MAX(date) independently. _resolve_common_as_of finds the
# one date every source agrees on instead.
SNAPSHOT_TABLES = ("prices_daily", "technical_features", "model_predictions", "entry_exit", "explanations")

_state: dict = {}


def _build_rag_index() -> VectorIndex:
    fundamentals_snapshot = pd.read_csv(DATA_PROCESSED / "fundamentals_snapshot.csv", index_col=0)
    news_scored = pd.read_csv(DATA_PROCESSED / "news_scored.csv")
    risk_output = pd.read_parquet(DATA_PROCESSED / "risk_engine_output.parquet")
    explainability_predictions = pd.read_parquet(DATA_PROCESSED / "explainability_predictions.parquet")
    shap_values = pd.read_parquet(DATA_PROCESSED / "explainability_shap_values.parquet")
    technical = pd.read_parquet(DATA_PROCESSED / "technical_features.parquet")
    technical_latest = technical.sort_values("date").groupby("symbol").tail(1)

    documents = (
        build_fundamentals_documents(fundamentals_snapshot)
        + build_technical_documents(technical_latest)
        + build_news_documents(news_scored)
        + build_risk_documents(risk_output)
        + build_explanation_documents(explainability_predictions, shap_values, top_factors, n_factors=3)
    )

    index = VectorIndex(load_sentence_transformer_embedder())
    index.build(documents)
    return index


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["engine"] = get_engine()
    _state["rag_index"] = _build_rag_index()
    yield
    _state.clear()


app = FastAPI(title="Indian Equity AI API", lifespan=lifespan)


def get_db_engine():
    """FastAPI dependency; overridden in tests to avoid touching the real DB."""
    return _state["engine"]


def get_rag_index() -> VectorIndex:
    """FastAPI dependency; overridden in tests to avoid loading the real embedding model."""
    return _state["rag_index"]


class AskRequest(BaseModel):
    symbol: str
    query: str


class AnalysisRequest(BaseModel):
    company: str
    horizon: str = "5d"
    analysis_type: str = "full"


def _query_df(engine, sql: str, params: dict) -> pd.DataFrame:
    return pd.read_sql(sql, engine, params=params)


def _json_safe_dict(row: dict) -> dict:
    """Converts NaN to None: Starlette's JSONResponse uses allow_nan=False (NaN
    isn't valid RFC-8259 JSON), and this project's data legitimately contains NaN
    for metrics that don't apply to every row (e.g. win_rate/profit_factor/turnover
    are undefined for a pure buy-and-hold benchmark) -- returning them as `null`
    is correct, not a workaround for a hidden bug.

    Must operate on a plain dict, not a DataFrame column: assigning None into a
    float64 column silently coerces back to NaN (pandas has no way to hold None
    in a fixed-float-dtype column), so `df.where(cond, None)` alone would not
    actually fix anything.
    """
    return {k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in row.items()}


def _json_safe_records(df: pd.DataFrame) -> list[dict]:
    return [_json_safe_dict(row) for row in df.to_dict(orient="records")]


def _json_safe_row(row: pd.Series) -> dict:
    return _json_safe_dict(row.to_dict())


def _table_max_date(engine, symbol: str, table: str) -> date | None:
    # date(date) normalizes SQLite's stored value (which can be a bare "YYYY-MM-DD"
    # or a full "YYYY-MM-DD HH:MM:SS" timestamp string depending on how a given
    # table was populated) to a plain date string, so this MAX() and the <=
    # comparison in _row_at_or_before below always compare like-for-like -- without
    # this, a timestamped row could be lexicographically "greater than" a bare
    # same-day date string and get silently excluded.
    df = _query_df(
        engine, f"SELECT MAX(date(date)) AS max_date FROM {table} WHERE symbol = :symbol", {"symbol": symbol}
    )
    value = df["max_date"].iloc[0]
    if value is None or pd.isna(value):
        return None
    return pd.to_datetime(value).date()


def _resolve_common_as_of(engine, symbol: str):
    """The one snapshot date every populated source-table agrees on for this
    symbol (see SNAPSHOT_TABLES above). Sources with no rows for this symbol are
    dropped, not treated as a failure -- e.g. a symbol with no entry/exit signal
    yet shouldn't block overview/technical from resolving a snapshot.
    """
    per_source_dates = {table: _table_max_date(engine, symbol, table) for table in SNAPSHOT_TABLES}
    try:
        return resolve_snapshot(per_source_dates)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"No data available for symbol '{symbol}'")


def _row_at_or_before(engine, symbol: str, table: str, as_of: date) -> dict:
    df = _query_df(
        engine,
        f"SELECT * FROM {table} WHERE symbol = :symbol AND date(date) <= :as_of ORDER BY date DESC LIMIT 1",
        {"symbol": symbol, "as_of": as_of.isoformat()},
    )
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No {table} data for symbol '{symbol}' at or before {as_of}")
    row = _json_safe_row(df.iloc[0])
    row["snapshot_as_of"] = as_of.isoformat()
    return row


def _snapshot_row(engine, symbol: str, table: str) -> dict:
    """Fetches `table`'s row at the shared cross-table as_of date, and attaches
    lagging_sources when other tables are ahead of it -- the caller sees exactly
    which snapshot it got and whether anything is silently behind, rather than a
    contradictory mix of dates with no visible explanation.
    """
    snapshot = _resolve_common_as_of(engine, symbol)
    row = _row_at_or_before(engine, symbol, table, snapshot.as_of)
    if snapshot.lagging_sources:
        row["lagging_sources"] = snapshot.lagging_sources
    return row


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stocks")
def list_stocks(engine=Depends(get_db_engine)):
    df = _query_df(engine, "SELECT DISTINCT symbol FROM prices_daily ORDER BY symbol", {})
    return df["symbol"].tolist()


@app.get("/stocks/{symbol}/overview")
def overview(symbol: str, engine=Depends(get_db_engine)):
    return _snapshot_row(engine, symbol, "prices_daily")


@app.get("/stocks/{symbol}/technical")
def technical(symbol: str, engine=Depends(get_db_engine)):
    return _snapshot_row(engine, symbol, "technical_features")


@app.get("/stocks/{symbol}/snapshot")
def snapshot(symbol: str, engine=Depends(get_db_engine)):
    """Single source of truth for "what date does this symbol's dashboard
    currently reflect" -- one call instead of independently querying six routes
    that could otherwise report six different dates. Powers the dashboard's top
    "Analysis as of" badge.
    """
    result = _resolve_common_as_of(engine, symbol)
    days_stale = trading_days_stale(result.as_of)
    return {**result.to_dict(), "trading_days_stale": days_stale, "staleness_label": staleness_label(days_stale)}


@app.get("/stocks/{symbol}/history")
def history(symbol: str, days: int = 180, engine=Depends(get_db_engine)):
    """OHLCV + indicator time series for charting. `days` counts trading days
    (rows), not calendar days -- this project only has daily EOD data, not
    intraday/streaming prices (see PROJECT_PLAN.md Phase 12's intraday note).
    """
    df = _query_df(
        engine,
        "SELECT * FROM technical_features WHERE symbol = :symbol ORDER BY date DESC LIMIT :days",
        {"symbol": symbol, "days": days},
    )
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No history for symbol '{symbol}'")
    return _json_safe_records(df.sort_values("date"))


@app.get("/stocks/{symbol}/fundamentals")
def fundamentals(symbol: str, engine=Depends(get_db_engine)):
    df = _query_df(engine, "SELECT * FROM fundamentals_snapshot WHERE symbol = :symbol", {"symbol": symbol})
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No fundamentals for symbol '{symbol}'")
    return _json_safe_row(df.iloc[0])


@app.get("/stocks/{symbol}/news")
def news_for_symbol(symbol: str, engine=Depends(get_db_engine)):
    df = _query_df(
        engine, "SELECT * FROM news WHERE symbol = :symbol ORDER BY published_timestamp DESC", {"symbol": symbol}
    )
    if not df.empty:
        records = _json_safe_records(df)
        for record in records:
            record["source_mode"] = "precomputed"
        return records

    # Precomputed news is empty (this project's documented Yahoo outage during
    # original data collection left it that way permanently) -- try a live fetch
    # instead of showing "no data" forever even once the outage has long passed.
    articles, meta = fetch_with_fallback(symbol)
    if articles.empty:
        return []
    scored = add_sentiment(articles, _default_sentiment_classify_fn())
    records = _json_safe_records(scored)
    for record in records:
        record["source_mode"] = "live_fallback"
        record["retrieved_at"] = meta.get("retrieved_at")
    return records


def _model_entry_label(entry: dict | None) -> str:
    if not entry or not entry.get("training_data_cutoff_date"):
        return "unavailable"
    return f"training cutoff {entry['training_data_cutoff_date']}"


@app.get("/stocks/{symbol}/lineage")
def lineage(symbol: str, engine=Depends(get_db_engine)):
    """Old (precomputed-universe) path's Data & Model Lineage panel (website audit
    Sections 32-33) -- reuses the same Tier 1-4/model_output system as the live
    orchestrator's lineage (src.rag.documents / src.services.lineage), applied to
    this symbol's precomputed data instead of a live fetch.
    """
    snapshot = _resolve_common_as_of(engine, symbol)
    fundamentals_df = _query_df(engine, "SELECT * FROM fundamentals_snapshot WHERE symbol = :symbol", {"symbol": symbol})
    fundamentals_row = fundamentals_df.iloc[0] if not fundamentals_df.empty else None

    news_df = _query_df(
        engine, "SELECT * FROM news WHERE symbol = :symbol ORDER BY published_timestamp DESC LIMIT 1", {"symbol": symbol}
    )
    has_news = not news_df.empty
    news_tier = provider_tier(news_df.iloc[0]["source"]) if has_news else TIER_FINANCIAL_WEBSITE

    model_entry = load_latest()
    quantile_entry = load_latest(filename=QUANTILE_REGISTRY_FILENAME)

    def _fundamentals_field(name: str) -> str:
        if fundamentals_row is None or pd.isna(fundamentals_row.get(name)):
            return "unavailable"
        return str(fundamentals_row[name])

    return [
        {
            "category": "market_data",
            "source": "Yahoo Finance (relayed NSE EOD data)",
            "tier": TIER_FINANCIAL_WEBSITE,
            "tier_label": TIER_LABELS[TIER_FINANCIAL_WEBSITE],
            "as_of_or_period": snapshot.as_of.isoformat(),
        },
        {
            "category": "fundamentals_balance_sheet",
            "source": "Yahoo Finance (aggregated financial data)",
            "tier": TIER_FINANCIAL_WEBSITE,
            "tier_label": TIER_LABELS[TIER_FINANCIAL_WEBSITE],
            "as_of_or_period": _fundamentals_field("fiscal_year"),
        },
        {
            "category": "fundamentals_valuation",
            "source": "Yahoo Finance (aggregated financial data)",
            "tier": TIER_FINANCIAL_WEBSITE,
            "tier_label": TIER_LABELS[TIER_FINANCIAL_WEBSITE],
            "as_of_or_period": _fundamentals_field("valuation_as_of"),
        },
        {
            "category": "news",
            "source": str(news_df.iloc[0]["source"]) if has_news else "unavailable",
            "tier": news_tier,
            "tier_label": TIER_LABELS[news_tier],
            "as_of_or_period": str(news_df.iloc[0]["published_timestamp"]) if has_news else "unavailable",
        },
        {
            "category": "model_classifier",
            "source": "This project's own ML pipeline (global cross-company classifier)",
            "tier": TIER_MODEL_OUTPUT,
            "tier_label": TIER_LABELS[TIER_MODEL_OUTPUT],
            "as_of_or_period": _model_entry_label(model_entry),
        },
        {
            "category": "model_quantile",
            "source": "This project's own ML pipeline (quantile-regression price range)",
            "tier": TIER_MODEL_OUTPUT,
            "tier_label": TIER_LABELS[TIER_MODEL_OUTPUT],
            "as_of_or_period": _model_entry_label(quantile_entry),
        },
    ]


@app.get("/stocks/{symbol}/forecast")
def forecast(symbol: str, engine=Depends(get_db_engine)):
    return _snapshot_row(engine, symbol, "model_predictions")


@app.get("/stocks/{symbol}/risk")
def risk(symbol: str, engine=Depends(get_db_engine)):
    row = _snapshot_row(engine, symbol, "model_predictions")
    result = {"risk_score": row["risk_score"], "risk_label": row["risk_label"], "snapshot_as_of": row["snapshot_as_of"]}
    if "lagging_sources" in row:
        result["lagging_sources"] = row["lagging_sources"]
    return result


@app.get("/stocks/{symbol}/explanation")
def explanation(symbol: str, engine=Depends(get_db_engine)):
    return _snapshot_row(engine, symbol, "explanations")


@app.get("/stocks/{symbol}/entry-exit")
def entry_exit(symbol: str, engine=Depends(get_db_engine)):
    return _snapshot_row(engine, symbol, "entry_exit")


@app.get("/backtest")
def backtest(engine=Depends(get_db_engine)):
    df = _query_df(engine, "SELECT * FROM backtest_results", {})
    return _json_safe_records(df)


@app.post("/ai-analyst/ask")
def ai_analyst_ask(request: AskRequest, rag_index: VectorIndex = Depends(get_rag_index)):
    return answer_query(request.query, request.symbol, rag_index, k=4)


@app.get("/company/{identifier}/resolve")
def resolve_company(identifier: str):
    """Phase 15: resolves a ticker/name (any supported NSE company, not just the
    50-stock training universe) to a canonical symbol, without running any analysis.
    """
    try:
        resolved = resolve(identifier)
    except (CompanyNotResolvedError, AmbiguousCompanyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {
        "symbol": resolved.symbol,
        "name": resolved.name,
        "in_training_universe": resolved.in_training_universe,
        "resolved_by": resolved.resolved_by,
    }


@app.post("/analysis")
def run_analysis(request: AnalysisRequest):
    """Phase 26: the live "train globally, infer locally" pipeline -- resolves any
    supported company (in the 50-stock training universe or not), fetches its data
    on demand, computes the same feature schema training used, and applies the
    already-trained global model. Never retrains for this request.
    """
    try:
        return analyze_company(
            request.company, horizon=request.horizon, analysis_type=request.analysis_type
        )
    except (CompanyNotResolvedError, AmbiguousCompanyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001 -- live network/data-source failures surface as 502s, not fabricated results
        raise HTTPException(status_code=502, detail=f"Analysis failed: {exc}")
