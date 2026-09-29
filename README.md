# Indian Equity AI Intelligence Platform

AI/ML-powered research and decision-support platform for the Indian stock market. Combines market data, technical indicators, fundamentals, financial news/sentiment, corporate events, and macroeconomic data to produce probabilistic, explainable equity insights.

This is a **research and decision-support tool**, not a guaranteed prediction or autonomous trading system.

See:

- `SUMMARY.md` — the single living summary of the whole project (status, findings, known bugs)
- `Indian_Equity_AI_Project_Plan.md` — full project vision, architecture, and rationale
- `PROJECT_PLAN.md` — phased execution plan (current progress tracker)

## Architecture: train globally, infer locally

The platform is built around one principle: **a single global ML model is trained once on a
fixed universe of companies, and applied live to any supported company on request — the model
is never retrained per request.**

```
User enters a company (ticker or name, in the training universe or not)
        |
Security Resolver          src/security/resolver.py
  -> canonical symbol + "is this in the training universe?"
        |
Live Data Fetch             src/ingestion/{market_data,statements,news}.py
  -> OHLCV, fundamentals, news -- same functions the training pipeline uses
        |
Data Sufficiency Gate       src/data_quality/eligibility.py
  -> refuses inference (INSUFFICIENT_DATA) rather than feeding a partial feature row
        |
Feature Engineering         src/models/dataset.py (build_feature_table / build_latest_features)
  -> the *same* transform used at training time -- training and inference cannot silently diverge
        |
Global Model                src/models/{train_global,registry,predict}.py
  -> the one persisted, versioned model, loaded (not retrained) for this request
        |
Risk / Entry-Exit / SHAP / Fundamentals / News / RAG
        |
Structured analysis response, tagged with training_universe_member: true/false
```

- **Training universe:** the 50 NIFTY 50 constituents in `src/config.py`. This is what the
  currently-persisted global model (`data/models/`, via `python -m src.models.train_global`) was
  fit and validated on.
- **Analysis universe:** any NSE-listed company the resolver can resolve and the data-quality
  gate accepts — a strict superset of the training universe. Not being in the training universe
  is not a resolution failure; the response simply says so explicitly
  (`model_coverage.in_training_universe`) rather than implying the model was trained on that
  specific company.
- **Held-out-company generalization test** (`src/models/evaluate_holdout_companies.py`):
  companies excluded from training scored ROC-AUC ~0.525 vs ~0.512 for companies the model did
  see — both near coin-flip, consistent with this project's other honest "no demonstrated edge"
  findings (see `SUMMARY.md`). The architecture works technically; it is not evidence of a real
  predictive edge in either direction.
- **Live entry points:** `POST /analysis` and `GET /company/{id}/resolve` (`src/api/main.py`),
  and the "Analyze Any Company" tab in the Streamlit dashboard — both additive, alongside the
  original precomputed-50-stock routes/tabs, which are unchanged.

## Setup

```bash
# Activate the existing venv (Windows)
venv\Scripts\activate

pip install -r requirements.txt
```

## Run tests

```bash
pytest
```

## Run the app

```bash
python -m src.models.train_global     # persist the global model to data/models/ (once, or after a retrain)
python -m src.db.load_data            # populate data/app.db from data/processed/
uvicorn src.api.main:app --reload     # API on :8000
streamlit run app/dashboard.py        # dashboard on :8501
```

## Data licensing & commercial-use notes

This project is currently for **educational/research use only**. Before any commercial or
production deployment, review:

- **Price/fundamentals/news data (`yfinance` → Yahoo Finance):** relayed, not licensed for
  redistribution. Yahoo's terms generally prohibit commercial redistribution of their data
  without a separate license. None of this project's price/fundamentals/news documents are
  tagged as an official Tier 1 source (see `src/rag/documents.py`) for exactly this reason —
  everything is Yahoo-relayed (Tier 3) or this project's own derived model output.
- **Direct NSE/BSE/SEBI data:** not currently used. Official Bhavcopy/filings/regulatory data
  would need separate data-sharing/licensing agreements with the exchange or regulator before
  commercial use (see `Indian_Equity_AI_Project_Plan.md` Section 5).
- **ML models used (FinBERT, sentence-transformers/all-MiniLM-L6-v2):** open-weight models with
  permissive licenses at the time of writing; verify current license terms before commercial
  redistribution or deployment.
- **Rate limits:** the ingestion code (`src/ingestion`, `src/fundamentals`, `src/news`) makes
  sequential, unauthenticated requests to Yahoo Finance. It has no built-in backoff/rate-limit
  handling — scaling well beyond ~50 stocks or running frequently would need that added.

## Platform positioning

Per `Indian_Equity_AI_Project_Plan.md` Section 42, this platform should be described as an
**AI-powered Indian Equity Research and Decision-Support Platform**, not an autonomous trading
system. Every model output in this codebase (probability, price range, risk label,
entry/exit zone, AI Analyst answer) is a model-generated estimate, not a guaranteed prediction
or investment advice — see the disclaimer surfaced in the dashboard (`app/dashboard.py`,
`DISCLAIMER`) and the AI Analyst's answer composer (`src/rag/analyst.py`, `DISCLAIMER`).

If this system is ever extended to give personalized investment advice or operate in a
SEBI-regulated context, obtain qualified legal/regulatory advice first — this project does not
constitute that advice.

## Limitations (honest, not hidden)

Full detail and the reasoning behind each is in `SUMMARY.md`; in short:

- **No demonstrated predictive edge.** Every phase that tested for it — including the held-out-
  company generalization test above — found next-day/short-term direction sits at ~0.50 ROC-AUC,
  and a realistic backtest loses money net of transaction costs.
- **No intraday support.** Only daily EOD data; a genuine intraday model would need a different,
  likely paid, real-time data source.
- **Thin, recent-only news coverage.** Yahoo's free news feed is a small rolling window, not a
  historical archive.
- **Fundamentals coverage varies by sector.** Bank/NBFC statements don't report EBIT, so
  ROCE/current-ratio/interest-coverage are left `null`, not fabricated.
- **Quantile price-range forecasts are under-calibrated** (~64% empirical coverage against an
  ~80% target) — read as relative-uncertainty guidance, not a literal confidence interval.
- **No licensed Tier-1 data.** All price/fundamentals/news data is Yahoo-relayed (Tier 3), not
  fetched directly from NSE/BSE/SEBI — see "Data licensing" above.
