"""Phase 26 orchestrator: analyze_company() runs the full live pipeline (resolve ->
fetch -> features -> global model -> risk -> entry/exit -> fundamentals -> news/
sentiment -> RAG -> explainability) for ANY supported company, not just the 50-stock
training universe -- the "train globally, infer locally" architecture this project's
generalization work is building toward.

Never retrains: uses whatever global model is currently registered
(src.models.registry / src.models.train_global). If no model has been persisted yet,
this raises loudly (src.models.predict.NoTrainedModelError) rather than fabricating
a prediction.
"""

# IMPORTANT (Windows): torch must be imported before pandas/pyarrow anywhere in this
# process, or torch's native DLL load fails with `OSError: [WinError 1114] ...
# c10.dll` -- see src/news/sentiment.py for the full explanation. This module is a
# plausible first import (scripts, tests, the FastAPI app), so it guards the same
# way src/api/main.py does.
import torch  # noqa: F401,E402

import functools  # noqa: E402
from typing import Optional  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data_quality.eligibility import assess_market_data_eligibility
from src.entryexit.engine import compute_entry_exit_validated
from src.explainability.confidence import confidence_label, prediction_confidence
from src.features.risk import add_drawdown
from src.features.risk import beta as compute_beta
from src.features.risk import max_drawdown as compute_max_drawdown
from src.fundamentals.ratios import add_growth_metrics, build_annual_fundamentals, snapshot_valuation_and_ownership
from src.fundamentals.score import fundamental_score
from src.ingestion.market_data import download_daily_ohlcv, download_index_ohlcv
from src.models.dataset import FEATURE_COLUMNS, build_latest_features
from src.models.predict import NoTrainedModelError, load_model, predict_proba_up
from src.models.predict_quantiles import NoTrainedQuantileModelError, load_quantile_models, predict_price_quantiles
from src.news.providers.chain import fetch_with_fallback
from src.risk.engine import compute_risk_score, risk_label
from src.risk.var_es import historical_expected_shortfall, historical_var
from src.security.resolver import resolve
from src.services.context import build_analysis_context
from src.services.lineage import build_lineage_panel
from src.technical.indicators import add_technical_indicators
from src.technical.score import technical_score as compute_technical_score
from src.validation.signal import classify_signal
from src.validation.snapshot import assert_single_snapshot

NIFTY_INDEX_TICKER = "^NSEI"
SUPPORTED_HORIZONS = {"5d", "60d"}

DISCLAIMER = (
    "Research/educational analytics output, not investment advice or a guaranteed "
    "prediction. This project's global model has, under honest walk-forward "
    "evaluation, shown no demonstrated predictive edge (ROC-AUC ~= 0.50) and no edge "
    "after realistic transaction costs -- see SUMMARY.md."
)


@functools.lru_cache(maxsize=1)
def _default_embed_fn():
    from src.rag.retrieval import load_sentence_transformer_embedder

    return load_sentence_transformer_embedder()


@functools.lru_cache(maxsize=1)
def _default_sentiment_classify_fn():
    from src.news.sentiment import load_finbert_pipeline

    return load_finbert_pipeline()


def _nn(value):
    """Converts NaN/None to None; leaves everything else as a plain float/str."""
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and pd.isna(value):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    return value


def _fetch_indicator_frame(symbol: str, period: str = "5y") -> pd.DataFrame:
    ohlcv = download_daily_ohlcv(symbol, period=period)
    # Live downloads can include a trailing row for today's still-open session: real
    # open/volume but no close yet (the session hasn't finished). That row isn't
    # usable data -- not "missing" to be imputed, just not printed yet -- so it's
    # dropped rather than fed into indicators as a fabricated/NaN latest close.
    ohlcv = ohlcv[ohlcv["close"].notna()].reset_index(drop=True)
    indicators = add_technical_indicators(ohlcv)
    indicators["technical_score"] = compute_technical_score(indicators)
    return indicators


def _compute_risk_block(indicator_df: pd.DataFrame) -> dict:
    """Best-effort: beta needs a live NIFTY fetch, which can fail independently of
    everything else. A failure degrades risk_score to unavailable, never a fabricated
    number -- matching compute_risk_score's own missing-input handling.
    """
    try:
        df = add_drawdown(indicator_df)
        returns = df["close"].pct_change().dropna()
        latest_volatility = df["close"].pct_change().rolling(20).std().iloc[-1]
        max_dd = compute_max_drawdown(df).iloc[0]
        var_95 = historical_var(returns)
        es_95 = historical_expected_shortfall(returns)

        try:
            index_df = download_index_ohlcv(NIFTY_INDEX_TICKER, period="5y")
            index_returns = index_df.set_index("date")["close"].pct_change()
            stock_returns = df.set_index("date")["close"].pct_change()
            beta_value = compute_beta(stock_returns, index_returns)
        except Exception:
            beta_value = float("nan")

        risk_row = pd.DataFrame(
            [
                {
                    "expected_volatility": latest_volatility,
                    "beta": beta_value,
                    "max_drawdown": max_dd,
                    "var_95": var_95,
                }
            ]
        )
        score = compute_risk_score(risk_row).iloc[0]
        label = risk_label(pd.Series([score])).iloc[0] if pd.notna(score) else None

        return {
            "available": True,
            "expected_volatility": _nn(latest_volatility),
            "beta": _nn(beta_value),
            "max_drawdown": _nn(max_dd),
            "var_95": _nn(var_95),
            "expected_shortfall_95": _nn(es_95),
            "risk_score": _nn(score),
            "risk_label": None if label is None else str(label),
        }
    except Exception as exc:
        return {"available": False, "reason": str(exc)}


def _compute_fundamentals_block(symbol: str) -> dict:
    """Website audit Sections 7-9: fiscal-year balance-sheet ratios and today's
    valuation snapshot are two different "as of" bases (a fiscal year-end vs
    today) -- conflating them into one flat dict with no period labels was the
    gap. balance_sheet_basis and valuation_basis are now kept distinct, each
    carrying its own as-of/period field.
    """
    try:
        from datetime import date

        from src.fundamentals.statements import download_statements

        statements = download_statements(symbol)
        annual = add_growth_metrics(build_annual_fundamentals(statements))
        if annual.empty:
            return {"available": False, "reason": "No financial statements returned."}
        latest = annual.iloc[-1]
        valuation = snapshot_valuation_and_ownership(statements["info"])

        row = pd.DataFrame(
            [
                {
                    "roe": latest.get("roe"),
                    "debt_to_equity": latest.get("debt_to_equity"),
                    "revenue_growth_yoy": latest.get("revenue_growth_yoy"),
                    "pe_ratio": valuation.get("pe_ratio"),
                }
            ]
        )
        score = fundamental_score(row).iloc[0]
        fiscal_year = str(annual.index[-1])

        balance_sheet_basis = {
            "fiscal_year": fiscal_year,
            "roe": _nn(latest.get("roe")),
            "roce": _nn(latest.get("roce")),
            "debt_to_equity": _nn(latest.get("debt_to_equity")),
            "current_ratio": _nn(latest.get("current_ratio")),
            "interest_coverage": _nn(latest.get("interest_coverage")),
            "revenue_growth_yoy": _nn(latest.get("revenue_growth_yoy")),
        }
        valuation_basis = {
            "as_of": date.today().isoformat(),
            **{key: _nn(val) for key, val in valuation.items()},
        }

        result = {
            "available": True,
            "balance_sheet_basis": balance_sheet_basis,
            "valuation_basis": valuation_basis,
            "fundamental_score": _nn(score),
            "note_on_bases": (
                f"fundamental_score blends a {fiscal_year} fiscal-year metric (ROE) with a "
                f"today's-snapshot metric (P/E) -- see balance_sheet_basis vs valuation_basis "
                f"for which fields come from which as-of date."
            ),
        }
        if pd.isna(latest.get("roce")):
            result["note"] = (
                "ROCE/current ratio/interest coverage are not meaningful for banks/NBFCs "
                "(no EBIT or current-asset split in their statements) and are left null, not "
                "fabricated -- consistent with this project's documented HDFCBANK handling."
            )
        return result
    except Exception as exc:
        return {"available": False, "reason": str(exc)}


def _compute_news_block(symbol: str, classify_fn=None) -> dict:
    """Uses the provider-fallback chain (website audit Section 28) rather than
    calling Yahoo directly -- a future second provider plugs in without this
    function changing. provider/retrieved_at are always included (even when no
    news was found) so the response's lineage can cite exactly when and from
    where this was fetched.
    """
    try:
        from src.news.sentiment import add_sentiment

        articles, news_meta = fetch_with_fallback(symbol)
        if articles.empty:
            return {
                "available": True,
                "news_count": 0,
                "sentiment": "unavailable",
                "articles": [],
                "note": (
                    "No recent news returned by any configured provider. Yahoo Finance's free "
                    "news feed only covers a small recent rolling window, not a historical archive."
                ),
                **news_meta,
            }
        classify_fn = classify_fn or _default_sentiment_classify_fn()
        scored = add_sentiment(articles, classify_fn)
        return {
            "available": True,
            "news_count": int(len(scored)),
            "avg_positive_probability": _nn(scored["positive_probability"].mean()),
            "avg_negative_probability": _nn(scored["negative_probability"].mean()),
            "articles": scored[
                ["headline", "source", "published_timestamp", "positive_probability", "negative_probability"]
            ].to_dict(orient="records"),
            **news_meta,
        }
    except Exception as exc:
        return {"available": False, "reason": str(exc)}


def _compute_rag_block(symbol: str, technical_row: pd.Series, fundamentals_block: dict, embed_fn=None) -> dict:
    try:
        from src.rag.analyst import answer_query
        from src.rag.documents import build_fundamentals_documents, build_technical_documents
        from src.rag.retrieval import VectorIndex

        tech_row_df = technical_row.to_frame().T.copy()
        tech_row_df["symbol"] = symbol
        docs = build_technical_documents(tech_row_df)

        balance_sheet = fundamentals_block.get("balance_sheet_basis") or {}
        valuation = fundamentals_block.get("valuation_basis") or {}
        if fundamentals_block.get("available") and all(
            balance_sheet.get(k) is not None for k in ("roe", "debt_to_equity", "revenue_growth_yoy")
        ) and valuation.get("pe_ratio") is not None:
            fdf = pd.DataFrame(
                [
                    {
                        "roe": balance_sheet["roe"],
                        "debt_to_equity": balance_sheet["debt_to_equity"],
                        "revenue_growth_yoy": balance_sheet["revenue_growth_yoy"],
                        "pe_ratio": valuation["pe_ratio"],
                        "fundamental_score": fundamentals_block["fundamental_score"],
                    }
                ],
                index=[symbol],
            )
            docs += build_fundamentals_documents(fdf)

        embed_fn = embed_fn or _default_embed_fn()
        index = VectorIndex(embed_fn)
        index.build(docs)
        summary = answer_query(f"Summarize the current outlook for {symbol}", symbol, index, k=5)
        return {"available": True, **summary}
    except Exception as exc:
        return {"available": False, "reason": str(exc)}


def _compute_explainability_block(entry: dict, model, feature_row: pd.DataFrame) -> dict:
    if model is None:
        return {
            "available": False,
            "reason": "The naive baseline was the validated winner; no feature-based explanation applies.",
        }
    if entry["model_name"] not in {"random_forest", "xgboost", "lightgbm"}:
        return {"available": False, "reason": f"SHAP TreeExplainer is not applicable to {entry['model_name']}."}
    try:
        from src.explainability.shap_explainer import build_explainer, explain_predictions, top_factors

        explainer = build_explainer(model)
        shap_df = explain_predictions(explainer, feature_row[FEATURE_COLUMNS])
        factors = top_factors(shap_df.iloc[0], n=5)
        return {
            "available": True,
            "positive_factors": {k: float(v) for k, v in factors["positive"].items()},
            "negative_factors": {k: float(v) for k, v in factors["negative"].items()},
            "units": "log-odds (SHAP TreeExplainer margin space, not probability points)",
        }
    except Exception as exc:
        return {"available": False, "reason": str(exc)}


def analyze_company(
    company_identifier: str,
    horizon: str = "5d",
    analysis_type: str = "full",
    model_version: Optional[str] = None,
) -> dict:
    """`horizon` is "5d" (swing) or "60d" (long-term) -- see src.models.horizons.
    `analysis_type` is "full" (default) or "quick" (skips fundamentals/news/RAG, the
    three slowest and heaviest optional blocks).
    """
    if horizon not in SUPPORTED_HORIZONS:
        raise ValueError(f"Unsupported horizon {horizon!r}; use one of {sorted(SUPPORTED_HORIZONS)}.")

    resolved = resolve(company_identifier)
    symbol = resolved.symbol

    indicator_df = _fetch_indicator_frame(symbol)
    eligibility = assess_market_data_eligibility(indicator_df)
    latest_row = indicator_df.sort_values("date").iloc[-1]

    response: dict = {
        "company": {
            "identifier": company_identifier,
            "symbol": symbol,
            "name": resolved.name,
            "training_universe_member": resolved.in_training_universe,
            "resolved_by": resolved.resolved_by,
        },
        "data_quality": eligibility.to_dict(),
        "model_coverage": {
            "in_training_universe": resolved.in_training_universe,
            "note": (
                "Included in the global model's training universe."
                if resolved.in_training_universe
                else "Not included in the original global training universe; this prediction "
                "was generated by applying the existing global cross-company model to freshly "
                "computed features for this company -- the model was NOT retrained for this request."
            ),
        },
        "technical": {
            "date": str(latest_row["date"].date()),
            "close": _nn(latest_row["close"]),
            "rsi_14": _nn(latest_row.get("rsi_14")),
            "macd_histogram": _nn(latest_row.get("macd_histogram")),
            "technical_score": _nn(latest_row.get("technical_score")),
            "price_vs_sma50": "above" if latest_row["close"] > latest_row.get("sma_50", np.nan) else "below",
            "price_vs_sma200": "above" if latest_row["close"] > latest_row.get("sma_200", np.nan) else "below",
        },
    }

    entry = None
    quantile_entry = None

    if not eligibility.ml_eligible:
        response["prediction"] = {
            "available": False,
            "reason": "INSUFFICIENT_DATA: " + (eligibility.reason or "feature vector incomplete"),
        }
        response["entry_exit"] = {"available": False, "reason": "No prediction available."}
        response["explainability"] = {"available": False, "reason": "No prediction available."}
    else:
        entry, model, scaler = load_model(model_version)

        latest_features = build_latest_features(indicator_df)
        feature_row = latest_features[latest_features["symbol"] == symbol].tail(1)
        # Safety net, not a fix for a known bug: both latest_row and feature_row
        # are derived from the one shared indicator_df fetch above, so their dates
        # must already agree. This guards against a future regression
        # reintroducing an independent fetch for either block (website audit's
        # core "single analysis snapshot" requirement).
        assert_single_snapshot(
            {
                "technical": pd.Timestamp(latest_row["date"]).date(),
                "features": pd.Timestamp(feature_row["date"].iloc[0]).date(),
            }
        )

        proba = predict_proba_up(feature_row, entry, model, scaler)
        probability_up = float(proba.iloc[0])
        confidence = float(prediction_confidence(pd.Series([probability_up])).iloc[0])
        conf_label = str(confidence_label(pd.Series([confidence])).iloc[0])
        signal = classify_signal(probability_up, confidence)

        response["prediction"] = {
            "available": True,
            "horizon": horizon,
            "served_horizon_days": entry.get("horizon_days", 1),
            "horizon_note": (
                "This global model is trained on a 1-trading-day-ahead target; multi-day "
                "horizon selection is a planned but not-yet-implemented feature (see "
                "src.models.horizons) -- read the probability below as a next-day estimate "
                "regardless of the horizon requested."
                if entry.get("horizon_days", 1) == 1 and horizon != "5d"
                else None
            ),
            "probability_up": probability_up,
            "probability_down": 1 - probability_up,
            "confidence": confidence,
            "confidence_label": conf_label,
            "signal_label": signal.label,
            "model_version": entry["model_version"],
            "model_name": entry["model_name"],
            "feature_schema_version": entry["feature_schema_version"],
        }

        try:
            quantile_entry, quantile_models = load_quantile_models()
            price_quantiles = predict_price_quantiles(
                feature_row, pd.Series([latest_row["close"]], index=feature_row.index), quantile_models
            )
            response["prediction"]["price_quantiles_available"] = True
            response["prediction"]["price_q10"] = _nn(price_quantiles["q10"].iloc[0])
            response["prediction"]["price_q50"] = _nn(price_quantiles["q50"].iloc[0])
            response["prediction"]["price_q90"] = _nn(price_quantiles["q90"].iloc[0])
            response["prediction"]["quantile_model_version"] = quantile_entry["model_version"]
        except NoTrainedQuantileModelError as exc:
            response["prediction"]["price_quantiles_available"] = False
            response["prediction"]["price_quantiles_reason"] = str(exc)

        ee = compute_entry_exit_validated(
            close=pd.Series([latest_row["close"]]),
            atr=pd.Series([latest_row.get("atr_14", np.nan)]),
            probability_up=pd.Series([probability_up]),
        ).iloc[0]
        response["entry_exit"] = {
            "available": True,
            "has_signal": bool(ee["has_signal"]),
            "entry_low": _nn(ee["entry_low"]),
            "entry_high": _nn(ee["entry_high"]),
            "stop": _nn(ee["stop"]),
            "target": _nn(ee["target"]),
            "valid": bool(ee["is_valid"]),
            "invalid_reason": ee["invalid_reason"],
            "note": "Model-generated hypothetical levels, not guaranteed prices.",
        }
        response["explainability"] = _compute_explainability_block(entry, model, feature_row)

    response["risk"] = _compute_risk_block(indicator_df)

    if analysis_type == "full":
        fundamentals_block = _compute_fundamentals_block(symbol)
        news_block = _compute_news_block(symbol)
        rag_block = _compute_rag_block(symbol, latest_row, fundamentals_block)
    else:
        skipped = {"available": False, "reason": "Skipped (analysis_type='quick')."}
        fundamentals_block = news_block = rag_block = skipped

    response["fundamentals"] = fundamentals_block
    response["sentiment"] = news_block
    response["rag"] = rag_block
    response["sources"] = rag_block.get("sources", [])
    response["disclaimer"] = DISCLAIMER

    news_meta = {"provider": news_block.get("provider"), "retrieved_at": news_block.get("retrieved_at")}
    context = build_analysis_context(
        identifier=company_identifier,
        resolved=resolved,
        latest_row=latest_row,
        entry=entry,
        quantile_entry=quantile_entry,
        fundamentals_block=fundamentals_block,
        news_meta=news_meta,
    )
    response["context"] = context.to_dict()
    response["lineage"] = build_lineage_panel(context, entry, quantile_entry)
    return response
