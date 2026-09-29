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
