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

- [x] Compute returns: daily, weekly, monthly, 5-day, 20-day, 60-day
- [x] Compute rolling volatility (10/20/30/60-day), max drawdown, beta
- [x] Compute correlation vs NIFTY 50 and sector peers

**DoD:** Stock analytics report/notebook produced for all 5 stocks. ✅ Met — `notebooks/02_returns_and_risk.ipynb` produces `data/processed/stock_analytics_report.csv` (drawdown, beta vs NIFTY 50, weekly/monthly return volatility) and `correlation_matrix.csv` for all 5 stocks. Sector-index correlation covers TCS/INFY (`^CNXIT`) and HDFCBANK (`^NSEBANK`); ITC's and RELIANCE's Yahoo Finance sector indices (`^CNXFMCG`, `^CNXENERGY`) return near-empty history and are skipped with a note in the notebook — NIFTY 50 and cross-stock correlation are still complete for all 5.

---

## Phase 3 — Technical Analysis Engine

- [x] Trend indicators: SMA (20/50/100/200), EMA (20/50)
- [x] Momentum indicators: RSI, MACD, ROC, Stochastic Oscillator, ADX
- [x] Volatility indicators: ATR, Bollinger Bands, Bollinger Band Width
- [x] Volume indicators: Volume MA, Volume Ratio, OBV, VWAP
- [x] Combine into a single Technical Score (0–100)

**DoD:** Reusable technical-features module in `src/technical/`, validated against at least one known reference calculation per indicator. Met — `src/technical/{trend,momentum,volatility,volume,indicators,score}.py`, each indicator checked against a hand-computed or analytically-derived reference value in `tests/test_technical.py` (15 tests). `notebooks/03_technical_features.ipynb` computes the full set for all 5 stocks and saves `data/processed/technical_features.parquet`.

---

## Phase 4 — Fundamental Analysis Engine

- [x] Ingest income statement, balance sheet, cash flow data for the 5 companies
- [x] Compute ratios: P/E, P/B, P/S, EV/EBITDA, ROE, ROCE, Debt/Equity, Current Ratio, Interest Coverage, Dividend Yield
- [x] Compute growth metrics (revenue/EPS/FCF growth and CAGR) and ownership data (promoter/FII/DII holding)
- [x] Combine into a Fundamental Score (0–100)

**DoD:** Fundamentals module in `src/fundamentals/` producing scores for all 5 stocks from raw filings/statements. Met — `src/fundamentals/{statements,ratios,score}.py`, 10 tests in `tests/test_fundamentals.py` with hand-computed reference values. `notebooks/04_fundamental_features.ipynb` produces `data/processed/fundamentals_snapshot.csv` (score range 45.5–74.5, zero NaN) and `fundamentals_annual.parquet` for all 5 stocks.

Data-source notes:
- Bank income statements (HDFCBANK) don't report EBIT/EBITDA or a current-assets/liabilities split, so ROCE/current ratio/interest coverage are `NaN` for it by design — the Fundamental Score itself only needs ROE/Debt-Equity/revenue growth/P·E, all of which are available for every stock.
- Ownership data uses Yahoo's `heldPercentInsiders`/`heldPercentInstitutions` as proxies for promoter/institutional holding — these are not split into FII/DII as NSE/BSE disclosures do, and pledged-share data isn't available from this source.

---

## Phase 5 — Baseline ML Models

- [x] Build ML dataset with target `next_day_direction` from Phase 1–3 features
- [x] Use strict chronological splits — no random shuffling, no look-ahead leakage
- [x] Train/evaluate progressively: naive baseline → Logistic Regression → Random Forest → XGBoost → LightGBM

**DoD:** Baseline model evaluated with classification metrics (accuracy, precision, recall, F1, ROC-AUC, log loss, Brier score); written note on what worked, what didn't, and which features mattered. Met — `src/models/{dataset,baseline,evaluate}.py` (13 tests), `notebooks/05_baseline_ml.ipynb` trains all 5 models with a 70/15/15 chronological split (5,210 rows, all 5 stocks pooled). Validation ROC-AUC clusters near 0.50 (naive 0.50, logistic/RF/XGBoost/LightGBM all in the ~0.47–0.52 range) — expected for price/technical-only next-day direction, and a useful sanity check that nothing is leaking (a dramatically higher score would be the red flag). Top features by importance: `atr_pct`, `return_5d`, `volatility_20d`, `rsi_14`, `price_to_sma20/50`. Full written reflection is in the notebook's final markdown cell.

---

## Phase 6 — News Intelligence & Sentiment (NLP)

- [x] Build news ingestion + deduplication + company/entity resolution pipeline
- [x] Apply FinBERT for financial sentiment (positive/neutral/negative probabilities — not a single label)
- [x] Classify events (earnings, dividend, M&A, debt, management change, etc.)
- [x] Estimate historical event impact per event type (e.g. average effect at +1d/+5d/+20d)

**DoD:** News/sentiment/event features generated and joinable to the daily feature table by (date, symbol), respecting actual publication timestamps. Met — `src/news/{ingestion,dedup,sentiment,events,impact,features}.py` (18 tests), `notebooks/06_news_sentiment.ipynb` runs the full pipeline end-to-end on real (non-fabricated) Yahoo Finance news for all 5 stocks: 12 articles survived dedup + entity-mention filtering, FinBERT scored real positive/neutral/negative probabilities, the keyword classifier tagged real event types (debt, earnings, dividend, management_change, product_launch, contract_win), and `aggregate_daily_news_features` produces a `(date, symbol)`-keyed table saved to `data/processed/news_daily_features.parquet`.

Data-source limitation: Yahoo's free news feed is a small recent rolling window (~10 items/ticker, last few days), not a historical archive, so the event-impact table (`news_event_impact_summary.csv`) is a real but small-sample methodology demonstration, not a statistically reliable estimate — `mean_return_20d` is correctly `NaN` for events too recent to have 20 trading days of forward price history yet.

Windows-specific bug found and fixed: `torch` (via `transformers`) fails to load (`WinError 1114`, `c10.dll`) if pandas/pyarrow are imported first in the process — pyarrow's bundled Arrow C++ runtime conflicts with torch's native DLLs. Fix: `import torch` must be the first heavy import in any process calling `load_finbert_pipeline()`; documented in `src/news/sentiment.py` and applied as the first cell of the notebook.

---

## Phase 7 — Multimodal ML Model

- [x] Combine price, technical, fundamental, news/sentiment, macro, and market-regime features into one feature vector
- [x] Build the market regime engine (bull/bear × high/low volatility classification) as an additional feature
- [x] Train an ensemble combining per-modality models with a **learned** meta-model (not fixed weights)
- [x] Validate hypotheses H1–H5 using walk-forward validation

**DoD:** Multimodal ensemble outperforms the Phase 5 baseline on held-out chronological test data. Partially met, reported honestly rather than oversold — see below.

Built: `src/macro/{ingestion,features}.py`, `src/regime/classifier.py` (point-in-time-safe, expanding-median volatility threshold), `src/models/{multimodal_dataset,ensemble,walkforward}.py` (18 new tests, 73 total). `notebooks/07_multimodal_model.ipynb` merges Phases 3/4/6 + new macro/regime data into 2,530 rows across all 5 stocks (2024-05-30 to 2026-09-24 — fundamentals' fiscal-year-plus-reporting-lag requirement is what shrinks the usable window vs. Phase 5's wider range), trains a stacking ensemble (per-modality LightGBM base models + a learned logistic-regression meta-model fit on validation-set predictions), and runs 4 expanding-window walk-forward folds.

**Result, stated plainly:** on the single train/val/test split, the multimodal ensemble does beat the technical-only baseline (ROC-AUC 0.531 vs 0.466) — satisfying the DoD's literal comparison. But the walk-forward view (4 folds, more trustworthy than one split) shows the *opposite*: technical-only alone scores best (mean ROC-AUC 0.538) while the ensemble (simple-average across folds, 0.509) and every other individual modality (fundamental 0.495, macro_regime 0.499, news exactly 0.500) sit at or below base rate. The single-split "win" looks like a favorable-window artifact rather than a real effect — reported honestly rather than cherry-picked. Root cause: fundamentals/news/macro_regime are currently weak-to-uninformative modalities (data-availability limits from Phases 4/6), so stacking them with a genuinely useful technical model mostly adds noise. Full numbers and per-hypothesis (H1–H5) discussion are in the notebook's final reflection cell.

Real bug found and fixed during this phase: the initial `FUNDAMENTAL_FEATURE_COLUMNS` included `roce`/`current_ratio`/`interest_coverage`, which are `NaN` for every HDFCBANK fiscal year (banks don't report EBIT) — the dataset's single `dropna` silently excluded HDFCBANK entirely (100%) before this was caught and fixed to reuse Phase 4's already-solved universally-available column set.

---

## Phase 8 — Volatility, Probabilistic Forecasting & Risk Engine

- [x] Build a separate volatility model
- [x] Extend direction predictions to probabilistic quantile forecasts (10th/50th/90th percentile range)
- [x] Build the Risk Engine: expected return, probability, expected volatility, max drawdown, beta, liquidity, VaR, Expected Shortfall → Low/Medium/High risk label

**DoD:** Every prediction ships with a probability, a range, and a risk label — not just a point estimate. Met — every row of `data/processed/risk_engine_output.parquet` has `probability_up`, a `price_q10`/`price_q50`/`price_q90` range, and a `risk_label`.

Built: `src/risk/{volatility_target,quantiles,var_es,engine}.py` (17 new tests, 90 total) — forward realized-volatility labeling (hand-verified window alignment), LightGBM quantile regression with monotonicity enforcement (independently trained quantile models aren't guaranteed non-crossing), historical VaR/Expected Shortfall, and a composite 0–100 risk score. `notebooks/08_risk_engine.ipynb` wires these together with Phase 2's beta/max-drawdown and Phase 5's classifier.

**Two honest calibration findings, not glossed over:**
- The volatility model's MAE came out statistically indistinguishable from a naive "assume the training-set average volatility continues" forecast — it isn't yet demonstrably adding value.
- The quantile price range's empirical 10–90 coverage was ~64% against an ~80% target — the range is currently too narrow and should be read as relative-uncertainty guidance, not a literal confidence interval, until recalibrated.

**Bug found and fixed:** the risk label initially came out 100% "Medium" for every test row — `compute_risk_score`'s absolute 0–100 caps (5% daily vol, beta=2, 60% drawdown, 5% VaR) were calibrated for a broader, more volatile universe than these 5 large-cap blue chips, so no row ever reached the Low/High ends. Fixed by adding `risk_label_relative`, which buckets using tertile cutoffs learned from the training split's score distribution (not test, avoiding leakage) — now spans all three labels sensibly (e.g. RELIANCE/HDFCBANK skew Low/Medium given shallower historical drawdowns; INFY/TCS skew High given steeper ones).

---

## Phase 9 — Backtesting & Validation Framework

- [x] Build a dedicated backtesting engine recording signal/entry/stop/target/exit/PnL with transaction costs and slippage
- [x] Implement walk-forward validation (investigate purged cross-validation / embargo)
- [x] Compare against NIFTY Buy & Hold; compute Sharpe, Sortino, Calmar, max drawdown, win rate, profit factor
- [x] Validate H6 — confirm edge survives realistic transaction costs/slippage

**DoD:** Research-quality backtest report comparing model-driven signals vs benchmark, net of costs. Met.

Built: `src/backtesting/{engine,metrics}.py` (11 new tests), plus an `embargo_days` parameter added to Phase 7's `expanding_window_folds` (2 more tests, 102 total) — a 1-day-ahead target needs the last training row(s) before each fold's test window excluded, since their label depends on a price at/after the test boundary. `notebooks/09_backtesting.ipynb` runs the Phase 5-style technical classifier through 4 embargoed walk-forward folds (521 combined out-of-sample trading days, 2024-08-27 to 2026-09-24), simulates trading the signals (threshold 0.55, 15bps round-trip cost+slippage), and compares against NIFTY 50 buy & hold over the identical period.

**H6 result, stated plainly: no edge survives — the strategy loses money outright, and costs make it much worse.** Cumulative return: strategy net-of-cost **‑64.9%**, strategy gross (no costs) **‑25.7%**, NIFTY 50 buy & hold **‑7.8%** (the walk-forward test window happens to span a declining market). Every risk-adjusted metric agrees: Sharpe ‑2.65 (net) vs ‑0.24 (benchmark), max drawdown ‑65% (net) vs ‑16% (benchmark). Turnover was high (~51% of symbol-days had an active position), so a strategy with no genuine edge compounds costs quickly. This is consistent with every prior phase's finding (ROC-AUC ~0.50) — H6 isn't "a small edge partially survives costs," it's "there was no edge to begin with, and acting on noise anyway is actively harmful, especially after costs." Reported plainly rather than reframed positively.

---

## Phase 10 — Explainable AI

- [x] Add SHAP-based per-prediction explanations (top positive/negative factors)
- [x] Surface confidence/uncertainty alongside every prediction

**DoD:** System can answer "why did the model make this prediction?" for any given stock/day with a factor breakdown. Met.

Built: `src/explainability/{shap_explainer,confidence}.py` (7 new tests, 107 total). `notebooks/10_explainability.ipynb` explains every test-set prediction (785 rows) with top-4 positive/negative SHAP factors plus a confidence score (0=coin flip, 1=certain).

Important unit note, verified by test not just asserted: SHAP values from `TreeExplainer` are in **log-odds (margin) units**, not probability points (`sigmoid(sum(shap_values) + expected_value) == predict_proba`) — factor contributions are reported and labeled as such rather than mislabeled as percentage points for a nicer-looking display.

*Update (found while building Phase 11):* the saved `explainability_predictions.parquet` and `explainability_shap_values.parquet` didn't share row alignment — the first was saved with a fresh reset index, the second kept its original pre-reset index — so reloading them separately in a later notebook would have silently misaligned every row. Fixed by resetting both to a shared index before saving, and Phase 10 was re-executed to regenerate the corrected files.

Confidence distribution came out fairly balanced (257 Low / 299 Medium / 229 High out of 785) — not overwhelmingly skewed toward "unsure" despite the near-0.50 ROC-AUC found in every prior phase. That's expected, not a contradiction: confidence measures how strongly the model leans, not whether the lean is correct. Cross-check: the top global SHAP features (`price_to_sma200`, `return_5d`, `volatility_60d`, `atr_pct`, `price_to_sma50`) substantially overlap with Phase 5's tree-split-based importances — two independent importance measures broadly agreeing is a reassuring pipeline-consistency signal.

---

## Phase 11 — LLM + RAG Layer

- [x] Build RAG index over company filings, financial results, news, corporate announcements, and the system's own model outputs
- [x] Respect the Tier 1–4 source reliability hierarchy (NSE/BSE/SEBI/filings > Reuters/Bloomberg > financial websites/analyst reports > blogs/forums/social)
- [x] Wire the LLM to explain quantitative outputs/evidence — it must not invent its own predictions
- [x] Support example AI Analyst queries (e.g. "Why is X bullish?", "What are the major risks?"), each answer traceable to a source

**DoD:** AI Analyst answers grounded, sourced questions about a given stock using retrieved evidence + model outputs. Met.

**Key decision (asked the user, since it materially changes the architecture and needs no invented capability):** no LLM API key is configured in this environment, and downloading a real generative model for local CPU inference would be slow and low quality. Per the user's choice, the "LLM" step is a **deterministic template composer**, not a generative model call — it can only restate retrieved document text plus its source/tier, so it structurally cannot invent facts or predictions (Section 32's core requirement is satisfied by construction, not by prompting). `answer_query`'s interface is decoupled from *how* answers are composed, so a real LLM (behind a strict "cite only what's retrieved" system prompt) can be swapped in later without touching retrieval/tiering.

Built: `src/rag/{documents,retrieval,analyst}.py` (14 new tests, 122 total) — real `sentence-transformers` (all-MiniLM-L6-v2) embeddings, an in-memory cosine-similarity vector index (pgvector deferred to Phase 13's Postgres app) with tier-based trust-weighted re-ranking, and the template answer composer. `notebooks/11_llm_rag.ipynb` builds a 32-document corpus from Phases 4/6/8/10's real outputs and answers 4 example queries across 3 stocks.

**Honest tiering, not inflated:** nothing is tagged Tier 1 (official NSE/BSE/SEBI/filings), because nothing was fetched directly from those sources — this project's prices/fundamentals are Yahoo-Finance-relayed (Tier 3), news is tiered per-provider (Reuters → Tier 2; Simply Wall St./GuruFocus/TechCrunch → Tier 3), and this project's own predictions/SHAP explanations are a distinct, always-labeled "internal model output" category, never conflated with an independent source. The tier-weighting sanity check (notebook Step 4) confirms trust-weighting actually changes ranking order, not just cosmetic labels.

Retrieval worked well without any hardcoded query-intent routing — pure semantic search correctly surfaced the fundamentals document first for "strongest fundamental factors" and the actual news article first for "how has recent news affected this stock."

**Bug found and fixed:** `build_news_documents` used `row["event_types"] or "none detected"`, which prints the literal text "nan" for a missing value — pandas `NaN` is truthy in Python, so the `or` fallback never triggers. Fixed with an explicit `pd.isna()` check.

---

## Phase 12 — Entry/Exit Engine & Investor Horizons

- [x] Split modeling by horizon: intraday, swing/short-term, long-term — separate feature sets/models per horizon
- [x] Build the entry/exit decision-support layer (entry zone, stop zone, target zone, risk label) — clearly labeled as hypothetical model-generated levels, not guarantees

**DoD:** Horizon-specific outputs and entry/exit suggestions available for at least the swing/short-term horizon. Met.

**Intraday is honestly out of scope**: this project only has daily EOD data (Phase 1); Section 26's intraday feature list (1/5/15-minute bars) needs a different, likely paid, real-time/historical data source, not something to fabricate.

Built: `src/models/horizons.py` (generalizes Phase 5's `add_target` to any forward horizon, same NaN-safety fix), `src/entryexit/engine.py` (rule-based entry/stop/target zones from ATR + probability, long-only, consistent with Phase 9's backtest convention) — 6 new tests, 128 total. `notebooks/12_horizons_and_entry_exit.ipynb` builds two genuinely different horizons: **swing/short-term** (5-day forward direction, technical features) and **long-term** (60-day forward direction, point-in-time fundamentals via Phase 7's `merge_fundamentals_pit`) — "do not use the same model for every investment horizon" (Section 26) is satisfied with real, differently-sourced features, not just a relabeled copy.

Results: swing ROC-AUC 0.505, long-term ROC-AUC 0.453 — both near base rate, consistent with every prior phase's finding. This doesn't confirm H2 (fundamentals matter more long-term) here, but is reported plainly as inconclusive-with-current-data rather than reframed positively — Phase 4/7's known fundamentals coverage gaps (only ~5 fiscal years, coarse point-in-time granularity) are the more likely explanation than "fundamentals don't matter." Entry/exit zones (1.5×ATR stop, 2:1 reward:risk target) compute correctly and sanely (stop below close, target above, entry zone tight around close) on the swing model's signals — a presentation/risk-management layer on top of existing forecasts, not a new source of edge.

---

## Phase 13 — Application & Dashboard

- [x] Build the FastAPI backend serving predictions/risk/explanations
- [x] Set up PostgreSQL persistence (schema: companies, securities, prices_daily, prices_intraday, financial_statements, fundamentals, ratios, corporate_actions, announcements, news, news_sentiment, news_events, macro_data, technical_features, model_predictions, backtest_results, model_versions)
- [x] Build the Streamlit dashboard: Overview / Technical / Fundamentals / News / Sentiment / AI Forecast / Risk / Backtest / AI Analyst tabs

**DoD:** End-to-end flow works — search a stock in the dashboard, see a live consolidated analysis backed by the pipelines from Phases 1–12. Met — verified live in a real browser (screenshots + full-page text extraction), not just unit tests: searched HDFCBANK, saw a real AI Outlook (Bullish, 62.5% P(up)), a real Risk label (Medium, 44/100), and a real cited AI Analyst answer to "What are the strongest fundamental factors?" that correctly surfaced the fundamentals document first.

**Key decision (asked the user):** PostgreSQL 14 is installed on this machine but wasn't running as a service, with no `psycopg2`/`sqlalchemy` set up. Standing up a live Postgres service is a bigger infrastructure step than this phase needs. Per the user's choice, persistence is **SQLite via SQLAlchemy Core** — same relational schema, no service to manage, and portable to Postgres later (swap the connection URL; no column types used are SQLite-specific).

Built:
- `src/db/{schema,load_data}.py` — 8-table schema (a real subset of Section 7, restricted to data this project actually collected — no empty placeholder tables for `prices_intraday`/`corporate_actions` etc.) and an ETL loader from `data/processed/*` into `data/app.db`
- `src/api/main.py` — FastAPI backend, 10 endpoints (`/stocks`, `/stocks/{symbol}/{overview,technical,fundamentals,news,forecast,risk,explanation,entry-exit}`, `/backtest`, `/ai-analyst/ask`), dependency-injected DB engine + RAG index (`Depends()`, not a bare global) so tests can override both without touching the real DB or loading the real embedding model
- `app/{api_client,dashboard}.py` — a thin, unit-tested HTTP client and the Streamlit dashboard (9 tabs matching Section 34's mockup), presentation-only with no model/DB logic of its own

14 new tests across `db`/`api`/`api_client` (142 total).

*Update: added an interactive price chart with indicator overlays.* The Overview tab now has a candlestick + volume chart (`app/charts.py`, pure/testable, no Streamlit dependency) with a multiselect for overlays (SMA 20/50/200, EMA 20/50, Bollinger Bands) and oscillator subplots (RSI, MACD), backed by a new `GET /stocks/{symbol}/history` endpoint. Labeled honestly in the UI as daily EOD data, not streaming intraday ticks — this project never collected intraday data (Phase 12's documented scope limit). Widened the `technical_features` DB table/ETL to carry OHLCV + EMA/Bollinger columns needed for charting (they existed in the source parquet all along, just weren't selected into the DB). 10 new tests (152 total).

Two real bugs found and fixed while wiring this together:
- **SQLite in-memory test isolation:** FastAPI runs sync route handlers in a threadpool, and plain `sqlite:///:memory:` gives each new connection its own empty database — tests intermittently saw "no such table" errors. Fixed with `StaticPool` (the standard SQLAlchemy pattern for sharing one in-memory DB across connections), applied to both the API and ETL test fixtures.
- **NaN breaks JSON serialization:** the `/backtest` endpoint 500'd in the live browser test (caught by actually using the app, not just unit tests) — Starlette's `JSONResponse` uses `allow_nan=False`, and `backtest_results` legitimately has `NaN` for metrics that don't apply to a pure buy-and-hold benchmark (win_rate, profit_factor). First fix attempt (`df.where(cond, None)`) silently failed because assigning `None` into a `float64` column just coerces back to `NaN` — pandas can't hold `None` in a fixed-float-dtype column. Real fix: sanitize NaN→None on the plain Python dicts *after* `to_dict()`, not on the DataFrame.

Also carried forward: `import torch` before pandas/pyarrow (Phase 6's finding) applies process-wide, not per-file — pytest runs all test modules in one process, so relying on individual files to each import torch first only works by alphabetical luck. Fixed properly with a root `tests/conftest.py` that imports torch before pytest collects anything.

---

## Phase 14 — Scale-Out

- [x] Expand universe: 5 stocks → 50 stocks → full selected NSE universe
- [x] Revisit data-source licensing/redistribution terms (API terms, rate limits, commercial-use restrictions) before any commercial-facing use
- [x] Position/document the platform as a research/decision-support tool — no guaranteed returns/predictions/signals claims; obtain legal/regulatory advice before any SEBI-regulated personalized advice use case

**DoD:** Pipelines proven on 5 stocks now run unmodified (or with clearly isolated changes) across the expanded universe. Met.

**Scope, per the user's explicit choice:** full re-run of every phase's notebook (not just config + a partial proof) on a 50-stock universe.

**Universe:** `src/config.py` centralizes `UNIVERSE` — previously hardcoded independently in 6 notebooks (01, 02, 03, 04, 06, 11), now a one-line import in each (`from src.config import UNIVERSE`), which is itself the "clearly isolated change" the DoD allows — no other pipeline code changed to support scale. Chose NIFTY 50 (a principled, well-defined "~50 liquid stocks" selection, not an arbitrary pick), with every ticker individually verified to resolve on Yahoo Finance before committing to the list. Two substitutions from the textbook NIFTY 50 list, both due to real 2025 corporate actions discovered during verification: Tata Motors demerged into passenger/commercial-vehicle entities (`TATAMOTORS` no longer resolves; used `TMPV`, the passenger-vehicle successor), and `LTIM` did not resolve via this data source (substituted `DMART`).

**Full re-run results — all 8 data-dependent notebooks (01, 02, 03, 04, 06, 07, 08, 09, 10, 11, 12) re-executed clean for 50 stocks, zero unhandled errors:**
- Phase 1 (prices): 50/50 stocks, consistent 2021-09-29→2026-09-28 date range — succeeded with **zero code changes** beyond the config import, the cleanest possible proof of the DoD.
- Phase 3 (technical): NaN-warm-up count scaled exactly (9,950 = 199×50), confirming no drift in the indicator math.
- Phase 4 (fundamentals): 47/50 stocks (94%) got a full Fundamental Score; 3 (`NESTLEIND`, `TMPV`, `SBILIFE`) didn't, reported via a warning, not a crash.
- Phase 5/7/8/9 (ML/multimodal/risk/backtest): results stayed consistent with the 5-stock findings — near-base-rate ROC-AUC, H6's "no edge survives costs" conclusion held.
- Phase 12 (horizons): the 60-day long-term horizon model scored ROC-AUC 0.570 on 50 stocks (vs 0.453 on 5) — a single train/val/test split, not walk-forward-validated here, so this is reported as an interesting data point worth a proper walk-forward re-test (Phase 9-style), not evidence of a confirmed edge.
- DB/API/dashboard: ETL reloaded all 8 tables at 50-stock scale; live API spot-check confirmed `/stocks` returns all 50 symbols and `/stocks/RELIANCE/overview` returns real data.

**A confirmed external outage, handled honestly, not faked:** partway through this re-run, Yahoo Finance's news endpoint (`finance.yahoo.com/xhr/ncp`) started returning HTTP 500 for every symbol tested — including `AAPL`/`MSFT`, confirming it's a Yahoo-side outage, not specific to this project or its 50-stock scale. News/RAG results for this run reflect 0 articles across all 50 stocks; this is documented as an external, transient limitation, not silently hidden or backfilled with fabricated data.

**Three real bugs found and fixed during this scale-out:**
- **`resolve_entity_mentions` column-loss bug:** with 0 input rows (a direct consequence of the Yahoo news outage above, but the bug itself is a real, general-purpose defect, not scale-specific), the `mentions_company` column inferred as `float64` instead of `bool` (an empty Python list has no dtype to infer from), and boolean-indexing a DataFrame with a non-bool empty mask silently drops every column, not just filters rows. Fixed by explicitly constructing the column as `pd.Series(..., dtype=bool)`. Regression test added (`tests/test_news.py`).
- **F-string escaping bug introduced while fixing the above:** a `\n` intended as an escape sequence inside an f-string (for cosmetic blank-line spacing) arrived as a literal newline byte after passing through a bash heredoc, producing a Python `SyntaxError` when the notebook cell executed. Fixed by avoiding embedded `\n` in string literals entirely (two separate `print()` calls) and verified by compiling every code cell before re-running.
- **Phase 4 hard-assert too brittle for 50 stocks:** the original notebook asserted 100% of stocks must have a computable Fundamental Score, which would abort the entire run over a single stock's data gap. With 5 hand-picked large caps this never triggered; across 50 (including newly-demerged `TMPV`) it would have. Changed to a per-stock `try/except` around the fetch loop plus a warning report for stocks with no computable score — visible, not silent, and no longer fatal.

**Licensing/positioning:** addressed in `README.md`'s new "Data licensing & commercial-use notes" and "Platform positioning" sections — none of this project's data is Tier 1 (official), everything is Yahoo-relayed or the project's own derived output, and the research/decision-support (not investment-advice) framing is consistent across the dashboard, AI Analyst, and README.

**Test suite:** `src/config.py` + `tests/test_config.py` (universe validity), `src/news/dedup.py` fix + regression test — 3 new tests, 155 total.

---

## Phase 15 — Generalization: Train Globally, Infer Locally

- [x] Resolve any supported NSE company (ticker, `.NS`/`.BO`, or name), not just the 50-stock training universe
- [x] Persist a trained global model (previously only ever fit inside a notebook, never saved)
- [x] Gate live inference on real data sufficiency instead of feeding a partial feature row
- [x] Build a live orchestrator that fetches, computes features, predicts, and explains for any company on demand, without retraining
- [x] Validate the "train on one set of companies, infer on another" claim with a held-out-company test

**DoD:** A company deliberately excluded from training can be analyzed end-to-end (resolve → live fetch → same feature schema as training → the already-trained global model → risk/entry-exit/SHAP), explicitly flagged as not-in-training-universe, without retraining the model. Met — verified live via `POST /analysis` and the dashboard's "Analyze Any Company" tab for both a training-universe company (TCS) and a deliberately-excluded one (DIXON Technologies); identical code path, differing only in the `training_universe_member` flag.

**Core principle:** the global model is trained once on the 50-stock universe and applied live to any company on request — never retrained per request. Analysis Universe (any resolvable NSE company) is a strict superset of Training Universe (the 50 stocks `train_global.py` fits on).

Built:
- `src/security/resolver.py` — ticker/`.NS`/`.BO`/company-name → canonical symbol, with a verified alias map for the 50 NIFTY constituents; unresolved tickers are returned as a plausible analysis-universe candidate, not a resolution failure
- `src/data_quality/eligibility.py` — refuses inference (`INSUFFICIENT_DATA`) when a company's live-fetched history can't produce a complete feature row (SMA-200's ~260-trading-day warm-up)
- `src/models/{train_global,registry,predict}.py` — the missing persistence link: fits the same candidates notebook 05 always did, selects by validation ROC-AUC, and actually saves the winner (`joblib`) plus a JSON registry entry (training universe, feature schema version, validation metrics)
- `src/services/company_analysis.py::analyze_company()` — the live orchestrator: one fetch → shared feature snapshot → global model → risk/entry-exit/SHAP → best-effort fundamentals/news/RAG, each independently degrading to `"available": false` with a reason on failure, never fabricating a result
- `src/models/evaluate_holdout_companies.py` — splits `UNIVERSE`'s tickers (not dates) into train/holdout subsets; held-out companies scored ROC-AUC 0.525 vs 0.512 for seen companies — both near coin-flip, reinforcing (not contradicting) this project's existing "no demonstrated edge" finding, and confirming the architecture works technically without claiming a real predictive edge in either direction
- `POST /analysis`, `GET /company/{id}/resolve` (`src/api/main.py`) and the dashboard's "Analyze Any Company" tab — additive, alongside the original precomputed-50-stock routes/tabs

**Real bug found by running the live path against real data, not caught by a unit test:** a live `period="5y"` yfinance pull can include a trailing row for today's still-open session — real `open`/`volume`, `NaN` close — which silently NaN'd every downstream technical feature for the "latest" row. Fixed by dropping rows with no close before computing indicators.

**Test suite:** 36 new tests (resolver, eligibility, registry, predict, train_global, holdout-generalization, orchestrator, API, dashboard client) — 191 total.

---

## Phase 16 — Website Audit Remediation: Single-Snapshot Consistency, Calibrated Signals, Lineage

A user audit of the deployed dashboard found contradictory numbers (header close next to a forecast computed from a different price) and a mathematically "backwards" long-trade stop-loss. Investigation found and fixed the actual root cause, plus added several transparency/calibration features the audit requested.

- [x] Trace and fix the cross-table data-snapshot desync causing contradictory numbers
- [x] Gate directional labels ("Bullish"/"Bearish") on model confidence, not probability alone
- [x] Add a live price-quantile forecast to the orchestrator (previously only `probability_up`)
- [x] Add a news-provider abstraction with live-fallback for the permanently-empty precomputed News tab
- [x] Itemize backtest transaction costs and surface backtest underperformance prominently
- [x] Separate fiscal-year fundamentals from today's valuation snapshot, each with its own as-of date
- [x] Add a Data & Model Lineage panel

**DoD:** Every snapshot-dependent number traces to one shared, inspectable "as of" date; a confidently-wrong directional label can't reach the UI; negative backtest performance is disclosed prominently, not buried. Met — verified against the real, independently-confirmed data: `prices_daily`/`technical_features` dated 2026-09-28 vs `model_predictions`/`entry_exit` dated 2026-09-21 (a week apart, because the notebooks producing them were last re-run on different days) — root cause of both the reported price mismatch and the "backwards" stop (the stop-loss formula itself was confirmed mathematically correct throughout).

Built:
- `src/validation/{snapshot,entry_exit,signal,freshness}.py` — `resolve_snapshot()` finds the one date every populated source table agrees on; `validate_long_setup()` flags logically inconsistent entry/stop/target ordering as INVALID rather than hiding it; `classify_signal()` gates direction on confidence (explicitly flagged as an uncalibrated starting policy, not validated against real calibration data); `trading_days_stale()`/`staleness_label()` use a trading-calendar-aware freshness policy
- `src/api/main.py` — every snapshot-dependent route (`overview`, `technical`, `forecast`, `risk`, `entry-exit`, `explanation`) now serves the shared cross-table date instead of each independently picking its own; new `GET /stocks/{symbol}/snapshot` and `GET /stocks/{symbol}/lineage` routes; `/stocks/{symbol}/news` falls back to a live fetch when the precomputed table is empty
- `src/models/{train_quantiles,predict_quantiles}.py` — persists a quantile-regression model the same way `train_global.py` persists the classifier, reusing `src/risk/quantiles.py`'s existing fit/predict machinery; held-out 80%-interval coverage reported honestly (75.9% — different from, and somewhat better than, the original Phase 8 notebook model's ~64%, most likely due to different hyperparameters between the two independently-trained models; neither number is hidden or reconciled)
- `src/news/providers/{base,yahoo,chain}.py` — a `NewsProvider` interface with `YahooNewsProvider` as the only real implementation today and a `fetch_with_fallback()` chain; every-provider-failure and every-provider-empty both yield an empty result + recorded errors, never a fabricated sentiment
- `src/backtesting/engine.py::compute_itemized_transaction_costs()` — itemized brokerage/STT/exchange-charges/SEBI-turnover-fee/stamp-duty/GST/slippage breakdown (illustrative research-grade approximations, documented as such), ~16.9bps round-trip, close to the original flat ~15bps estimate; `turnover()` (which measured position-occupancy, not portfolio turnover) renamed `active_position_rate()`, kept as a deprecated alias
- `src/services/{context,lineage}.py` — `AnalysisContext` formalizes the live orchestrator's single-snapshot pattern into one inspectable object; `build_lineage_panel()` reuses `src/rag/documents.py`'s existing Tier 1–4/model_output source-reliability system — no new tiering logic invented
- `notebooks/04_fundamental_features.ipynb` re-run (scoped to this one notebook, not the full pipeline) to backfill `fiscal_year`/`valuation_as_of` columns on `fundamentals_snapshot`, separating fiscal-year balance-sheet ratios from today's valuation snapshot
- Dashboard: a top-of-page "Analysis as of" staleness badge, a prominent backtest-underperformance warning (net of costs: -56.2% vs -7.6% for buy-and-hold NIFTY 50), a Data & Model Lineage expander, and the live orchestrator's price-quantile range + horizon-disclosure note

**Two real bugs found by actually running the fix, not by unit tests:**
- SQLite stores a `date` column as either a bare `YYYY-MM-DD` or a full timestamp string depending on how a given table was populated; a bare-string `<=` comparison silently excluded same-day timestamped rows. Fixed with SQLite's `date()` function normalizing both sides of every comparison.
- The dashboard's own backtest-warning code picked `strategy_cols[0]` after alphabetically sorting non-benchmark columns, which selected `strategy_gross` (looks fine next to the benchmark) instead of `strategy_net_of_cost` (the actually-damning comparison) — would have silently suppressed the exact warning this phase was built to add. Fixed by naming the `strategy_net_of_cost` column explicitly.

**Also found during this phase, scoped to a disclosure-only fix:** requesting `horizon="60d"` silently returned the same 1-trading-day-ahead classifier prediction relabeled as a 60-day forecast (no horizon-specific model exists). The response now includes `served_horizon_days` and an explicit note; full multi-horizon model training is a legitimate separate future initiative, not undertaken here.

**Test suite:** 64 new tests (`src/validation/`, news-provider chain, quantile train/predict, `AnalysisContext`/lineage, itemized costs, and snapshot-consistency regression tests that directly mirror the real repo's confirmed data drift) — 255 total.

---

## How to use this file

Work top to bottom. Check off items as completed. Do not begin a phase's work until the previous phase's DoD is satisfied — this is the core discipline the source planning document calls out repeatedly (avoid building the whole system at once; each phase must leave you with something that actually runs).
