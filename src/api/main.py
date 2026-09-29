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
from pathlib import Path  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
from fastapi import Depends, FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from src.db.schema import get_engine  # noqa: E402
from src.explainability.shap_explainer import top_factors  # noqa: E402
from src.rag.analyst import answer_query  # noqa: E402
from src.rag.documents import (  # noqa: E402
    build_explanation_documents,
    build_fundamentals_documents,
    build_news_documents,
    build_risk_documents,
    build_technical_documents,
)
from src.rag.retrieval import VectorIndex, load_sentence_transformer_embedder  # noqa: E402
from src.security.resolver import (  # noqa: E402
    AmbiguousCompanyError,
    CompanyNotResolvedError,
    resolve,
)
from src.services.company_analysis import analyze_company  # noqa: E402

DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

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


def _latest_row(engine, symbol: str, table: str) -> dict:
    df = _query_df(
        engine, f"SELECT * FROM {table} WHERE symbol = :symbol ORDER BY date DESC LIMIT 1", {"symbol": symbol}
    )
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No {table} data for symbol '{symbol}'")
    return _json_safe_row(df.iloc[0])


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stocks")
def list_stocks(engine=Depends(get_db_engine)):
    df = _query_df(engine, "SELECT DISTINCT symbol FROM prices_daily ORDER BY symbol", {})
    return df["symbol"].tolist()


@app.get("/stocks/{symbol}/overview")
def overview(symbol: str, engine=Depends(get_db_engine)):
    return _latest_row(engine, symbol, "prices_daily")


@app.get("/stocks/{symbol}/technical")
def technical(symbol: str, engine=Depends(get_db_engine)):
    return _latest_row(engine, symbol, "technical_features")


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
    return _json_safe_records(df)


@app.get("/stocks/{symbol}/forecast")
def forecast(symbol: str, engine=Depends(get_db_engine)):
    return _latest_row(engine, symbol, "model_predictions")


@app.get("/stocks/{symbol}/risk")
def risk(symbol: str, engine=Depends(get_db_engine)):
    row = _latest_row(engine, symbol, "model_predictions")
    return {"risk_score": row["risk_score"], "risk_label": row["risk_label"]}


@app.get("/stocks/{symbol}/explanation")
def explanation(symbol: str, engine=Depends(get_db_engine)):
    return _latest_row(engine, symbol, "explanations")


@app.get("/stocks/{symbol}/entry-exit")
def entry_exit(symbol: str, engine=Depends(get_db_engine)):
    return _latest_row(engine, symbol, "entry_exit")


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
