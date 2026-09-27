# Phased Project Plan — Indian Equity AI Intelligence Platform

> Derived from `Indian_Equity_AI_Project_Plan.md`. Guiding principle from that document:
> **"Do not start with AI. Start with data."**
> DATA → UNDERSTANDING → FEATURES → BASELINE → MACHINE LEARNING → VALIDATION → NEWS/NLP → MULTIMODAL MODEL → RISK → EXPLAINABILITY → LLM → APPLICATION
>
> No calendar estimates here by design — each phase ends when its **Definition of Done (DoD)** is met, not on a fixed date. Work sequentially; do not start a phase until the prior phase's DoD is checked off.

---

## Phase 0 — Environment & Repo Setup

- [x] Set up Python, virtual environment, Jupyter, Git/GitHub repo
- [x] Create the recommended folder structure: `data/`, `notebooks/`, `src/`, `models/`, `tests/`, `app/`, `configs/`
- [x] Confirm/resolve the Flask-vs-FastAPI stack question in `requirement.txt` (resolved: FastAPI + Streamlit, per source doc; `requirement.txt` replaced by `requirements.txt`)

**DoD:** Repo scaffolded, dependencies installable, an empty pipeline runs end-to-end (no-op).

---

## Phase 1 — Historical Market Data

- [x] Start with RELIANCE only: download, load into Pandas, inspect, clean historical OHLCV
- [x] Calculate daily returns and rolling volatility; plot price; save cleaned dataset
- [x] Expand the same pipeline to the full 5-stock universe: RELIANCE, TCS, INFY, HDFCBANK, ITC (5 years daily OHLCV)

**DoD:** `01_stock_data_exploration.ipynb` runs clean for all 5 stocks; data validated (no missing/duplicate/invalid values) and saved to `data/processed/`. ✅ Met — 1,241 rows/symbol (2021-09-27 to 2026-09-25), zero data-quality issues on the cleaned set, saved as per-symbol parquet files plus a combined `prices_daily.parquet`.

---

## Phase 2 — Exploratory Data Analysis

- [ ] Compute returns: daily, weekly, monthly, 5-day, 20-day, 60-day
- [ ] Compute rolling volatility (10/20/30/60-day), max drawdown, beta
- [ ] Compute correlation vs NIFTY 50 and sector peers

**DoD:** Stock analytics report/notebook produced for all 5 stocks.

---

## Phase 3 — Technical Analysis Engine

- [ ] Trend indicators: SMA (20/50/100/200), EMA (20/50)
- [ ] Momentum indicators: RSI, MACD, ROC, Stochastic Oscillator, ADX
- [ ] Volatility indicators: ATR, Bollinger Bands, Bollinger Band Width
- [ ] Volume indicators: Volume MA, Volume Ratio, OBV, VWAP
- [ ] Combine into a single Technical Score (0–100)

**DoD:** Reusable technical-features module in `src/technical/`, validated against at least one known reference calculation per indicator.

---

## Phase 4 — Fundamental Analysis Engine

- [ ] Ingest income statement, balance sheet, cash flow data for the 5 companies
- [ ] Compute ratios: P/E, P/B, P/S, EV/EBITDA, ROE, ROCE, Debt/Equity, Current Ratio, Interest Coverage, Dividend Yield
- [ ] Compute growth metrics (revenue/EPS/FCF growth and CAGR) and ownership data (promoter/FII/DII holding)
- [ ] Combine into a Fundamental Score (0–100)

**DoD:** Fundamentals module in `src/fundamentals/` producing scores for all 5 stocks from raw filings/statements.

---

## Phase 5 — Baseline ML Models

- [ ] Build ML dataset with target `next_day_direction` from Phase 1–3 features
- [ ] Use strict chronological splits — no random shuffling, no look-ahead leakage
- [ ] Train/evaluate progressively: naive baseline → Logistic Regression → Random Forest → XGBoost → LightGBM

**DoD:** Baseline model evaluated with classification metrics (accuracy, precision, recall, F1, ROC-AUC, log loss, Brier score); written note on what worked, what didn't, and which features mattered.

---

## Phase 6 — News Intelligence & Sentiment (NLP)

- [ ] Build news ingestion + deduplication + company/entity resolution pipeline
- [ ] Apply FinBERT for financial sentiment (positive/neutral/negative probabilities — not a single label)
- [ ] Classify events (earnings, dividend, M&A, debt, management change, etc.)
- [ ] Estimate historical event impact per event type (e.g. average effect at +1d/+5d/+20d)

**DoD:** News/sentiment/event features generated and joinable to the daily feature table by (date, symbol), respecting actual publication timestamps.

---

## Phase 7 — Multimodal ML Model

- [ ] Combine price, technical, fundamental, news/sentiment, macro, and market-regime features into one feature vector
- [ ] Build the market regime engine (bull/bear × high/low volatility classification) as an additional feature
- [ ] Train an ensemble combining per-modality models with a **learned** meta-model (not fixed weights)
- [ ] Validate hypotheses H1–H5 using walk-forward validation

**DoD:** Multimodal ensemble outperforms the Phase 5 baseline on held-out chronological test data.

---

## Phase 8 — Volatility, Probabilistic Forecasting & Risk Engine

- [ ] Build a separate volatility model
- [ ] Extend direction predictions to probabilistic quantile forecasts (10th/50th/90th percentile range)
- [ ] Build the Risk Engine: expected return, probability, expected volatility, max drawdown, beta, liquidity, VaR, Expected Shortfall → Low/Medium/High risk label

**DoD:** Every prediction ships with a probability, a range, and a risk label — not just a point estimate.

---

## Phase 9 — Backtesting & Validation Framework

- [ ] Build a dedicated backtesting engine recording signal/entry/stop/target/exit/PnL with transaction costs and slippage
- [ ] Implement walk-forward validation (investigate purged cross-validation / embargo)
- [ ] Compare against NIFTY Buy & Hold; compute Sharpe, Sortino, Calmar, max drawdown, win rate, profit factor
- [ ] Validate H6 — confirm edge survives realistic transaction costs/slippage

**DoD:** Research-quality backtest report comparing model-driven signals vs benchmark, net of costs.

---

## Phase 10 — Explainable AI

- [ ] Add SHAP-based per-prediction explanations (top positive/negative factors)
- [ ] Surface confidence/uncertainty alongside every prediction

**DoD:** System can answer "why did the model make this prediction?" for any given stock/day with a factor breakdown.

---

## Phase 11 — LLM + RAG Layer

- [ ] Build RAG index over company filings, financial results, news, corporate announcements, and the system's own model outputs
- [ ] Respect the Tier 1–4 source reliability hierarchy (NSE/BSE/SEBI/filings > Reuters/Bloomberg > financial websites/analyst reports > blogs/forums/social)
- [ ] Wire the LLM to explain quantitative outputs/evidence — it must not invent its own predictions
- [ ] Support example AI Analyst queries (e.g. "Why is X bullish?", "What are the major risks?"), each answer traceable to a source

**DoD:** AI Analyst answers grounded, sourced questions about a given stock using retrieved evidence + model outputs.

---

## Phase 12 — Entry/Exit Engine & Investor Horizons

- [ ] Split modeling by horizon: intraday, swing/short-term, long-term — separate feature sets/models per horizon
- [ ] Build the entry/exit decision-support layer (entry zone, stop zone, target zone, risk label) — clearly labeled as hypothetical model-generated levels, not guarantees

**DoD:** Horizon-specific outputs and entry/exit suggestions available for at least the swing/short-term horizon.

---

## Phase 13 — Application & Dashboard

- [ ] Build the FastAPI backend serving predictions/risk/explanations
- [ ] Set up PostgreSQL persistence (schema: companies, securities, prices_daily, prices_intraday, financial_statements, fundamentals, ratios, corporate_actions, announcements, news, news_sentiment, news_events, macro_data, technical_features, model_predictions, backtest_results, model_versions)
- [ ] Build the Streamlit dashboard: Overview / Technical / Fundamentals / News / Sentiment / AI Forecast / Risk / Backtest / AI Analyst tabs

**DoD:** End-to-end flow works — search a stock in the dashboard, see a live consolidated analysis backed by the pipelines from Phases 1–12.

---

## Phase 14 — Scale-Out

- [ ] Expand universe: 5 stocks → 50 stocks → full selected NSE universe
- [ ] Revisit data-source licensing/redistribution terms (API terms, rate limits, commercial-use restrictions) before any commercial-facing use
- [ ] Position/document the platform as a research/decision-support tool — no guaranteed returns/predictions/signals claims; obtain legal/regulatory advice before any SEBI-regulated personalized advice use case

**DoD:** Pipelines proven on 5 stocks now run unmodified (or with clearly isolated changes) across the expanded universe.

---

## How to use this file

Work top to bottom. Check off items as completed. Do not begin a phase's work until the previous phase's DoD is satisfied — this is the core discipline the source planning document calls out repeatedly (avoid building the whole system at once; each phase must leave you with something that actually runs).
