# Indian Equity AI Platform — Project Summary

> **This is the single living summary of the whole project.** Update it whenever a phase
> changes, a new finding emerges, or a bug is found/fixed — it should always reflect current
> reality, not a point-in-time snapshot. For full phase-by-phase detail (checklists, DoD text,
> exact file lists), see `PROJECT_PLAN.md`. For the original vision/spec, see
> `Indian_Equity_AI_Project_Plan.md`.

**What this is:** an AI/ML research and decision-support platform for Indian equities —
price/technical/fundamental/news/macro data feeding probabilistic, explainable predictions,
served through a FastAPI backend and Streamlit dashboard with an AI Analyst.
**What this is not:** a guaranteed-returns system or investment advice.

**Status:** All 14 planned phases complete, the generalization effort (Phases 15–26, "train
globally, infer locally") complete, plus a website-audit remediation pass fixing a real,
independently-verified data-snapshot-consistency bug and adding confidence-calibrated signal
labels, live price-quantile forecasts, a news-provider abstraction, itemized backtest costs, and
a Data & Model Lineage panel — see below. Universe: 50 stocks (NIFTY 50), configured in
`src/config.py`. Test suite: 255/255 passing.

---

## The one finding that matters most

Every phase that tested for predictive edge (Phases 5, 7, 8, 9, 12) found next-day/short-term
direction from price, technical, fundamental, news, and macro features sits at **~0.50 ROC-AUC
— essentially no edge over a coin flip.** Phase 9's realistic backtest (transaction costs +
slippage) showed a strategy built on this signal **loses money outright** (‑64.9% net of cost
over the test window, vs ‑7.8% for simply holding NIFTY 50). This is reported consistently and
plainly throughout, not reframed as a partial success — it is itself a legitimate research
result (consistent with weak-form market efficiency), and the honest baseline this platform's
outputs should be read against.

One partial exception: Phase 12's 60-day long-term horizon model scored ROC-AUC 0.570 on the
50-stock universe (vs 0.453 on 5) — a single train/val/test split, **not yet walk-forward
validated**. Worth re-testing properly (Phase 9-style) before treating it as a real signal.

The held-out-company generalization test (Phase 22, below) reinforces the same finding from a
different angle: companies the model never saw during training (ROC-AUC 0.525) perform
statistically the same as companies it did see (0.512) — both near coin-flip. That is itself a
consistency check that the architecture works technically (no crash, no leakage, no degenerate
output), not evidence of a real edge in either direction.

A **real, independently-verified data bug** was found and fixed during the website-audit pass
(below): the precomputed dashboard's `model_predictions`/`entry_exit` tables were silently 7 days
stale relative to `prices_daily`/`technical_features` (different notebooks last re-run on
different days), producing exactly the contradictory header-vs-forecast prices and a
mathematically "backwards" stop-loss the audit reported. This was a data-pipeline bug, not a
finding about market predictability — see "Website audit remediation" below for the fix and how
it was verified against the real, reproducing data.

---

## Phase-by-phase

| # | Phase | Built | Key result |
|---|---|---|---|
| 0 | Environment & Repo Setup | Folder structure, `requirements.txt`, git repo, smoke test | Working env, FastAPI+Streamlit stack chosen |
| 1 | Historical Market Data | `src/ingestion/market_data.py` | 5→50 stocks, 5yr daily OHLCV, zero data-quality issues |
| 2 | Exploratory Data Analysis | `src/features/risk.py` | Returns, volatility, drawdown, beta, correlation vs NIFTY 50 |
| 3 | Technical Analysis Engine | `src/technical/*` | SMA/EMA/RSI/MACD/ATR/Bollinger/ADX/OBV/VWAP + 0–100 Technical Score, each hand-verified |
| 4 | Fundamental Analysis Engine | `src/fundamentals/*` | ROE/ROCE/Debt-Equity/growth/valuation + 0–100 Fundamental Score |
| 5 | Baseline ML Models | `src/models/{dataset,baseline,evaluate}.py` | LogReg/RF/XGBoost/LightGBM vs naive; ROC-AUC ~0.50 (no leakage, no edge) |
| 6 | News Intelligence & Sentiment | `src/news/*` | FinBERT sentiment + event classification on real Yahoo news |
| 7 | Multimodal ML Model | `src/macro/*`, `src/regime/*`, stacking ensemble | Ensemble "wins" on 1 split, **loses** under walk-forward — reported honestly |
| 8 | Volatility/Risk Engine | `src/risk/{volatility_target,quantiles,var_es,engine}.py` | Every prediction ships probability + price range + risk label |
| 9 | Backtesting & Validation | `src/backtesting/{engine,metrics}.py` | H6: no edge survives realistic costs (net ‑64.9% vs benchmark ‑7.8%) |
| 10 | Explainable AI | `src/explainability/{shap_explainer,confidence}.py` | SHAP factor breakdown + confidence per prediction |
| 11 | LLM + RAG Layer | `src/rag/{documents,retrieval,analyst}.py` | Grounded, cited AI Analyst (template composer, not generative — see below) |
| 12 | Entry/Exit & Horizons | `src/models/horizons.py`, `src/entryexit/engine.py` | Swing (5d) + long-term (60d) horizons; rule-based entry/stop/target zones |
| 13 | Application & Dashboard | `src/db/*`, `src/api/main.py`, `app/*` | FastAPI + SQLite + Streamlit (9 tabs), verified live in browser |
| 14 | Scale-Out | `src/config.py` | 5→50 stocks (NIFTY 50), full pipeline re-run, zero code drift |
| 15 | Security Resolver | `src/security/resolver.py` | Ticker/`.NS`/`.BO`/company-name → canonical symbol; distinguishes training-universe vs any-NSE-company |
| 18 | Data Sufficiency Gate | `src/data_quality/eligibility.py` | Refuses ML inference (`INSUFFICIENT_DATA`) rather than feeding a partial feature row |
| 21 | Global Model Persistence | `src/models/{train_global,registry,predict}.py` | First-ever **persisted** model artifact (previously notebook-only); XGBoost won validation, test ROC-AUC 0.507 |
| 22 | Held-Out-Company Test | `src/models/evaluate_holdout_companies.py` | 40/10 ticker split; unseen-company ROC-AUC 0.525 vs seen-company 0.512 — see finding above |
| 26 | Live Orchestrator | `src/services/company_analysis.py` | `analyze_company()`: resolve → fetch → features → global model → risk/entry-exit/SHAP → fundamentals/news/RAG, for any company |
| 28 | API Extension | `src/api/main.py` | `POST /analysis`, `GET /company/{id}/resolve` — additive, no existing route changed |
| 29 | Dashboard Extension | `app/dashboard.py` | New "Analyze Any Company" tab (free-text search) alongside the existing 50-stock dropdown |

---

## Generalization: train globally, infer locally

Phases 0–14 built a batch pipeline over a fixed 50-stock universe with no persisted model and no
live inference path — the FastAPI app (Phase 13) only ever read precomputed DB rows. Phases
15–26 close that gap without touching any of Phases 0–14's working code:

- `src/models/train_global.py` fits the same candidate models notebook 05 always did, but now
  actually **persists** the winner (`joblib`) plus a registry entry (`src/models/registry.py`,
  a JSON log, not a DB table — training is offline/batch, decoupled from the app's SQLite DB).
- `src/security/resolver.py` resolves a ticker, `.NS`/`.BO`-suffixed ticker, or company name to
  a canonical symbol, and flags whether it's in the training universe — not being in it is not a
  resolution failure (Analysis Universe ⊇ Training Universe).
- `src/data_quality/eligibility.py` gates inference on real feature-completeness (SMA-200's
  ~260-trading-day warm-up), returning `INSUFFICIENT_DATA` honestly rather than feeding the
  model a partial row.
- `src/services/company_analysis.py::analyze_company()` is the live orchestrator: resolve →
  fetch (on demand, any symbol) → the *same* `build_feature_table`/`build_latest_features`
  transform training used → the persisted global model → risk/entry-exit/SHAP → best-effort
  fundamentals/news/RAG (each independently wrapped, degrading to `"available": false` with a
  reason on failure, never fabricating a result). **The model is never retrained per request.**
- Verified end-to-end for both a training-universe company (TCS) and a deliberately-excluded one
  (DIXON Technologies) — both flow through identical code, differing only in the
  `training_universe_member` flag the response carries.
- One real bug found by actually running this against live data (see bug log below): yfinance's
  `period="5y"` pull can include a trailing row for the still-open trading session — real
  `open`/`volume` but `NaN` close. Filtered out in the live-fetch path, not imputed.
- `src/api/main.py` gained `POST /analysis` and `GET /company/{id}/resolve` (additive); the
  dashboard gained an "Analyze Any Company" tab (additive) alongside the existing 9 precomputed
  tabs. All 155 original tests plus 36 new ones (resolver, eligibility, registry, predict,
  train_global, holdout-generalization, orchestrator, API, dashboard client) pass — 191 total.

Not yet built (tracked, not started): sector-aware fundamental sub-scoring beyond the existing
bank-NaN handling, a persistent cache layer for repeated live lookups, and wiring a second
production model version through the registry to exercise `feature_schema_version` mismatch
handling for real (currently only unit-tested with a synthetic stale version).

---

## Website audit remediation: single-snapshot consistency, calibrated signals, lineage

A user audit of the deployed dashboard found several numbers that contradicted each other (e.g.
header close ₹2,831 next to a forecast range of ₹2,927–₹3,023) and a mathematically "backwards"
long-trade stop-loss (above the current price). Root-cause investigation found and fixed the
actual bug, plus added several transparency/calibration features the audit requested:

- **Root cause, confirmed and independently reproduced in the real repo data:**
  `src/api/main.py`'s per-table "latest date" queries (`prices_daily`/`technical_features` vs
  `model_predictions`/`entry_exit`) could silently diverge — verified directly: the former pair
  was dated 2026-09-28, the latter pair 2026-09-21, a week apart, because the notebooks producing
  them were last re-run on different days. `src/entryexit/engine.py`'s stop-loss formula was
  confirmed mathematically correct throughout (`stop = close - multiple*atr` is always `< close`
  for the same row) — the "backwards stop" was this same cross-table desync, not a formula bug.
- **Fix:** `src/validation/snapshot.py::resolve_snapshot()` finds the one date every populated
  source table agrees on; every snapshot-dependent API route (`overview`, `technical`, `forecast`,
  `risk`, `entry-exit`, `explanation`) now serves that shared date instead of each independently
  picking its own table's max date, and a new `GET /stocks/{symbol}/snapshot` route plus a
  dashboard "Analysis as of" badge make staleness visible rather than silently contradictory.
  Verified against the real, still-reproducing data: after the fix, `/overview` and `/forecast`
  for ADANIENT both consistently report `snapshot_as_of = 2026-09-21`, flagged `"very_stale"`.
- **Confidence-gated signal labels** (`src/validation/signal.py`): "Bullish"/"Bearish" previously
  came from `probability_up > 0.5` alone, ignoring model confidence entirely — a 57.9% probability
  with confidence 0.11 (near coin-flip) was shown as unqualified "Bullish". Now gated by the same
  thresholds `src/explainability/confidence.py` already defines: low confidence always reports
  "Low Confidence" regardless of direction. Thresholds are an explicitly-flagged, uncalibrated
  starting policy, not validated against real calibration data (consistent with the project's own
  quantile-calibration caution below).
- **Live price-quantile forecast**: `src/models/train_quantiles.py` persists a quantile-regression
  model (reusing `src/risk/quantiles.py`'s existing fit/predict machinery) the same way
  `train_global.py` persists the classifier, so the live orchestrator (`analyze_company()`) now
  also returns a `price_q10/q50/q90` range from the same snapshot as everything else. Honest
  result: this specific model's held-out 80%-interval coverage was **75.9%** — different from (and
  somewhat better than) the previously-documented Phase 8 notebook model's ~64%, most likely due
  to different hyperparameters/config between the two independently-trained models. Both numbers
  are below the 80% target; this is reported as-is, not reconciled or improved to match either
  prior figure.
- **Horizon-mislabeling bug found during this pass**: requesting `horizon="60d"` silently returned
  the same 1-trading-day-ahead classifier prediction relabeled as if it were a 60-day forecast.
  Scoped to a disclosure-only fix (not full multi-horizon model training, which is materially
  larger): the response now includes `served_horizon_days` and an explicit note when the
  requested horizon isn't actually what was modeled.
- **News provider abstraction** (`src/news/providers/`): a `NewsProvider` interface with
  `YahooNewsProvider` as the only real implementation today and a `fetch_with_fallback()` chain,
  so a future second provider plugs in without touching call sites. The old dashboard's News tab
  previously only ever read the permanently-empty precomputed `news` table (empty due to the
  already-documented Yahoo outage during original data collection, not a bug); it now falls back
  to a live fetch when the precomputed table is empty.
- **Itemized backtest transaction costs** (`src/backtesting/engine.py::compute_itemized_transaction_costs`):
  replaced the flat combined-bps assumption with an itemized brokerage/STT/exchange-charges/SEBI-
  turnover-fee/stamp-duty/GST/slippage breakdown (illustrative research-grade approximations,
  documented as such, not a live broker rate card) totaling ~16.9bps round-trip, close to the
  original flat ~15bps estimate. `turnover()` (which actually measured position-occupancy, not
  portfolio turnover) is renamed `active_position_rate()`, with `turnover` kept as a deprecated
  alias. A prominent dashboard warning now shows when the strategy underperforms its benchmark
  net of costs (it does: -56.2% vs -7.6% for buy-and-hold NIFTY 50 in the current data), instead
  of that only being visible inside the Backtest tab.
- **Fundamentals metadata**: `fundamentals_snapshot` previously had no period/source fields at
  all, conflating fiscal-year balance-sheet ratios (ROE, ROCE, Debt/Equity) with today's valuation
  snapshot (P/E, P/B) into one flat, undated row. `notebooks/04_fundamental_features.ipynb` was
  re-run (scoped to just that one notebook, not the full pipeline) to backfill `fiscal_year`/
  `valuation_as_of` columns for the 50-stock universe; the live orchestrator's fundamentals block
  now separates `balance_sheet_basis` from `valuation_basis` explicitly.
- **Data & Model Lineage panel**: a new `GET /stocks/{symbol}/lineage` route (old path) and
  `response["lineage"]` (live path) reuse `src/rag/documents.py`'s existing Tier 1–4/model_output
  source-reliability system — no new tiering logic invented — to show source/tier/as-of for market
  data, fundamentals (both bases), news, and both models, explicitly separating a model's training
  cutoff from the market data it's being applied to.
- Verified end-to-end in the browser against real, live-reproducing data (not just synthetic
  tests): the staleness badge, the backtest-underperformance warning, confidence-downgraded
  signal labels, the lineage table, and the live quantile range + horizon-disclosure note all
  render correctly for both a precomputed-universe symbol (ADANIENT) and a live "Analyze Any
  Company" query (DIXON, 60-day horizon).
- Test suite: 255/255 passing (191 → 255; new coverage for `src/validation/`, the news-provider
  chain, quantile train/predict, `AnalysisContext`/lineage, itemized costs, and snapshot-
  consistency regression tests that directly mirror the real repo's confirmed data drift).

---

## Key architectural decisions (asked of the user, not assumed)

- **LLM (Phase 11):** no API key configured → the "AI Analyst" is a **deterministic template
  composer**, not a generative model call. It can only restate retrieved, cited evidence, so it
  structurally cannot invent predictions. `answer_query()`'s interface is decoupled from *how*
  answers are composed, so a real LLM can be swapped in later behind a strict "cite only
  retrieved evidence" prompt.
- **Database (Phase 13):** PostgreSQL was installed but not running as a service → persistence
  is **SQLite via SQLAlchemy Core** instead. Same schema; portable to Postgres later by changing
  the connection URL only.
- **Scale-out scope (Phase 14):** the user chose a **full re-run of every phase's notebook**
  on the 50-stock universe, not just a config-only proof.

---

## Known data-source limitations (honest, not hidden)

- **Nothing is Tier-1 sourced.** All price/fundamentals/news data is Yahoo-Finance-relayed
  (Tier 3), not fetched directly from NSE/BSE/SEBI. This project's own model outputs are a
  distinct "internal model output" category, never conflated with an independent source. See
  `src/rag/documents.py` and `README.md`'s licensing section.
- **News coverage is thin.** Yahoo's free news feed is a small recent rolling window
  (~10 items/ticker), not a historical archive. During the Phase 14 scale-out, Yahoo's news
  endpoint had a **confirmed external outage** (HTTP 500 for every symbol, including AAPL/MSFT)
  — handled gracefully (0 articles, no crash), not faked.
- **Sector-index correlation is incomplete.** `^CNXFMCG`/`^CNXENERGY` (ITC/RELIANCE's sectors)
  return near-empty history on Yahoo; skipped with a note (Phase 2).
- **Fundamentals coverage varies.** Bank statements (HDFCBANK) don't report EBIT, so
  ROCE/current-ratio/interest-coverage are `NaN` for banks by design (Phase 4). At 50-stock
  scale, 47/50 got a full Fundamental Score; 3 didn't (reported via warning, not a crash).
- **Quantile forecast is under-calibrated.** Phase 8's original notebook model showed ~64%
  empirical coverage against an ~80% target; the persisted model now served live
  (`src/models/train_quantiles.py`) independently measured ~75.9% on its own held-out split —
  better, but still below target. Both numbers are reported as-is, not reconciled; read either as
  relative-uncertainty guidance, not a literal confidence interval, until recalibrated.
- **Intraday is out of scope.** This project only has daily EOD data; a genuine intraday model
  (Section 26) would need a different, likely paid, real-time data source (Phase 12).

---

## Real bugs found and fixed along the way

Each of these was caught by actually running the code/app (not just writing it), consistent with
this project's practice of verifying before declaring a phase done.

1. **`yfinance==0.2.50` auth handshake broken** → upgraded to `1.7.0` (Phase 1).
2. **`beta()` duplicate-column bug**: two Series with the same name collapsed into a DataFrame
   under `pd.concat`, breaking indexing (Phase 2).
3. **`technical_score` NaN-comparison bug**: `sma_50 > NaN` evaluates `False` in pandas, not
   `NaN` — silently faked a "downtrend" score during warm-up instead of `NaN` (Phase 3).
4. **Windows torch/pyarrow DLL conflict**: `import torch` must happen before
   pandas/pyarrow anywhere in the process, or native DLL loading fails (`WinError 1114`).
   Originally fixed per-file; later found to be **process-wide**, not per-file — pytest runs all
   test modules in one process, so a root `tests/conftest.py` was needed (Phases 6, 11, 13).
5. **`FUNDAMENTAL_FEATURE_COLUMNS` silently dropped HDFCBANK** (100%) via a `dropna` on
   bank-incompatible columns (Phase 7).
6. **Risk label was 100% "Medium"**: absolute scoring caps were calibrated for a more volatile
   universe than 5 blue-chips. Fixed with relative (train-tertile) labeling (Phase 8).
7. **`explainability_predictions.parquet` / `explainability_shap_values.parquet` index
   mismatch**: saved with different (unaligned) indices, would silently misalign every row on
   reload (found while building Phase 11).
8. **`event_types or "none detected"` printed literal "nan"**: pandas `NaN` is truthy in Python,
   so `or` never falls back (Phase 11).
9. **SQLite in-memory test isolation**: FastAPI's threadpool + plain `sqlite:///:memory:` gave
   each connection its own empty DB. Fixed with `StaticPool` (Phase 13).
10. **NaN breaks JSON serialization**: Starlette disallows `NaN`; first fix attempt
    (`df.where(cond, None)`) silently failed because `None` in a `float64` column coerces back
    to `NaN`. Real fix sanitizes on plain dicts after `to_dict()` (Phase 13).
11. **`resolve_entity_mentions` column-loss bug**: an empty `mentions_company` column inferred
    as `float64`, and boolean-indexing with a non-bool empty mask dropped every column (Phase 14).
12. **F-string `\n` escaping bug**: a literal newline arrived via a bash heredoc where an escape
    sequence was intended, causing a `SyntaxError` (Phase 14).
13. **Phase 4's hard assert too brittle for 50 stocks**: would abort the whole run over one
    stock's data gap; changed to per-stock `try/except` + visible warning (Phase 14).
14. **Trailing NaN-close row from live yfinance pulls**: a live `period="5y"` download can
    include today's still-open session — real `open`/`volume`, `NaN` close — which silently
    NaN'd every downstream technical feature for the "latest" row. Found by actually running the
    new live orchestrator against real data, not from a unit test; fixed by dropping rows with
    no close before computing indicators (Phase 26).
15. **Cross-table snapshot desync** (website audit): `prices_daily`/`technical_features` and
    `model_predictions`/`entry_exit` can silently drift apart in max date (confirmed 7 days apart
    in real data) since each API route picked its own table's latest date independently —
    produced the audit's reported header-vs-forecast price mismatch and "backwards" stop-loss.
    Fixed with a shared common-date resolver (`src/validation/snapshot.py`); see "Website audit
    remediation" above for full detail.
16. **Dashboard picked the wrong backtest column for the underperformance warning**: iterating
    `backtest_pivot.columns` and taking the first non-benchmark entry alphabetically selected
    `strategy_gross` (-4.5%, looks fine next to NIFTY's -7.6%) instead of `strategy_net_of_cost`
    (-56.2%, the actually-damning comparison) — would have silently suppressed the exact warning
    this pass was built to add. Found by manually verifying the feature in a browser against real
    data, not by a unit test; fixed by naming the `strategy_net_of_cost` column explicitly
    (website audit remediation).

---

## Tech stack

Python, pandas/numpy/pyarrow, scikit-learn/XGBoost/LightGBM, SHAP, PyTorch + Transformers
(FinBERT) + sentence-transformers (RAG embeddings), FastAPI + SQLAlchemy (SQLite) + Streamlit +
Plotly, pytest. Full list in `requirements.txt`.

## Where things live

- **Research notebooks:** `notebooks/01` through `notebooks/12`, one per phase — each is the
  authoritative, executed record of that phase's results.
- **Reusable pipeline code:** `src/` (mirrors the phase structure: `ingestion`, `features`,
  `technical`, `fundamentals`, `models`, `news`, `macro`, `regime`, `risk`, `backtesting`,
  `explainability`, `rag`, `entryexit`, `db`, `api`), the generalization additions `security`
  (resolver), `data_quality` (eligibility), and `services` (live orchestrator, `AnalysisContext`,
  lineage), and the website-audit additions `validation` (snapshot/entry-exit/signal/freshness)
  and `news/providers` (provider-fallback chain).
- **App:** `src/api/main.py` (FastAPI) + `app/dashboard.py` (Streamlit) + `app/{api_client,charts}.py`.
- **Tests:** `tests/`, 255 passing — hand-computed or analytically-derived reference values
  throughout, not just "does it run" checks.
- **Data:** `data/processed/*` (parquet/CSV, regenerated by the notebooks), `data/models/*`
  (persisted global classifier + quantile models + their registries, regenerated by
  `train_global.py`/`train_quantiles.py`), and `data/app.db` (SQLite, regenerated by
  `python -m src.db.load_data`) — all gitignored, not committed.

## Running it

```bash
venv\Scripts\activate
pip install -r requirements.txt
pytest                                    # 255 tests
python -m src.models.train_global         # persist the global classifier to data/models/
python -m src.models.train_quantiles      # persist the price-quantile models to data/models/
python -m src.db.load_data                # populate data/app.db from data/processed/
uvicorn src.api.main:app --reload         # API on :8000
streamlit run app/dashboard.py            # dashboard on :8501
```
