# AI-Powered Indian Equity Intelligence Platform

## Project Overview

This project aims to build an AI/ML-powered research and decision-support platform for the Indian stock market.

The platform will allow a new or inexperienced investor to search for an Indian listed company and receive a consolidated analysis covering:

- Current and historical stock prices
- Market and sector trends
- Technical analysis
- Fundamental/financial analysis
- Company news
- Financial-news sentiment
- Corporate events and announcements
- Historical price behavior
- Machine-learning-based probabilistic forecasts
- Risk analysis
- Explainable AI insights
- Different analysis horizons: intraday, swing/short-term, and long-term

The system is intended primarily as an **educational/research and decision-support platform**, not as a guaranteed stock-prediction or autonomous trading system.

---

# 1. Main Objective

Develop a multimodal machine-learning system that combines:

1. Market/price data
2. Technical indicators
3. Company fundamentals
4. Financial news
5. News sentiment
6. Corporate events
7. Macroeconomic indicators
8. Market regime information

to estimate the probability and expected range of future stock movements.

The system should also explain **why** the model reached its conclusion.

---

# 2. Proposed Research Question

> Can a multimodal machine-learning model combining market data, technical indicators, company fundamentals, financial-news sentiment, corporate events, and macroeconomic variables improve next-day directional forecasting of Indian equities compared with price-only and technical-analysis baselines?

## Research Hypotheses

### H1
Adding technical indicators improves next-day directional prediction compared with price-only models.

### H2
Adding fundamental variables provides additional predictive information for medium- and long-term analysis.

### H3
Financial-news sentiment and event information improve next-day directional prediction.

### H4
Market-regime features improve prediction stability across different market conditions.

### H5
A multimodal ensemble performs better than individual models under strict temporal/walk-forward validation.

### H6
The predictive improvement remains meaningful after accounting for transaction costs, slippage, and realistic trading assumptions.

---

# 3. High-Level System Architecture

```text
                         USER
                           |
                           v
                    Web Application
                           |
                           v
                       FastAPI
                           |
       +-------------------+-------------------+
       |                   |                   |
       v                   v                   v
   Market Data        Fundamentals           News
       |                   |                   |
       v                   v                   v
   Technical           Financial             NLP
   Features            Features            Features
       |                   |                   |
       +-------------------+-------------------+
                           |
                           v
                     Feature Store
                           |
                           v
                    ML Model Layer
                           |
        +------------------+------------------+
        |                  |                  |
        v                  v                  v
   Classification      Regression        Volatility
     Models              Models           Models
        |                  |                  |
        +------------------+------------------+
                           |
                           v
                     Ensemble Layer
                           |
                           v
                      Risk Engine
                           |
                           v
                  Explainability Layer
                           |
                           v
                       LLM + RAG
                           |
                           v
                    Investor Dashboard
```

---

# 4. Project Development Strategy

The project will be developed incrementally.

## Phase 1 — Historical Market Data

Start with only five stocks:

| Company | NSE Symbol | Sector |
|---|---|---|
| Reliance Industries | RELIANCE | Conglomerate |
| TCS | TCS | IT |
| Infosys | INFY | IT |
| HDFC Bank | HDFCBANK | Banking |
| ITC | ITC | FMCG |

Initial dataset:

- 5 years of historical daily data
- OHLCV
- Trading date
- Stock symbol

### Initial columns

```text
date
symbol
open
high
low
close
volume
```

### First objective

Build a reliable pipeline that can:

1. Download data
2. Validate data
3. Clean data
4. Calculate returns
5. Calculate volatility
6. Visualize prices
7. Save the processed data

---

# 5. Data Sources

## Primary / Official Sources

### NSE India

Use NSE for official Indian market information where appropriate.

Potential data:

- Historical/EOD market data
- Bhavcopy
- Corporate announcements
- Corporate actions
- Securities information
- F&O information
- Market reports

Official source:

https://www.nseindia.com/

### BSE India

Potential data:

- Historical prices
- Corporate announcements
- Corporate filings
- Company information

Official source:

https://www.bseindia.com/

### SEBI

Potential data:

- Regulatory filings
- Corporate disclosures
- Investment adviser/research analyst regulations
- Regulatory information

Official source:

https://www.sebi.gov.in/

---

## Prototype / Development Sources

For early experimentation, easier APIs/data providers can be used.

Potential sources:

- Yahoo Finance
- Alpha Vantage
- Financial Modeling Prep
- Zerodha Kite Connect
- Upstox API
- Angel One SmartAPI
- Groww API

### Important

Do not assume that data obtained from an API can automatically be redistributed in a commercial product.

Check:

- API terms
- licensing
- redistribution rights
- rate limits
- historical-data restrictions
- commercial-use restrictions

---

# 6. Data Architecture

Initial architecture:

```text
External Data Sources
        |
        v
Data Ingestion
        |
        v
Raw Data
        |
        v
Data Cleaning
        |
        v
Processed Data
        |
        v
PostgreSQL
        |
        v
Feature Engineering
        |
        v
ML Training Dataset
```

Later:

```text
Raw Data
    |
    +--> Object Storage / Data Lake
    |
    +--> PostgreSQL
    |
    +--> Feature Store
```

---

# 7. Database Design

Use PostgreSQL initially.

Suggested tables:

```text
companies
securities
prices_daily
prices_intraday
financial_statements
fundamentals
ratios
corporate_actions
announcements
news
news_sentiment
news_events
macro_data
technical_features
model_predictions
backtest_results
model_versions
```

## companies

```text
company_id
company_name
isin
sector
industry
exchange
```

## securities

```text
security_id
company_id
exchange
symbol
isin
```

## prices_daily

```text
date
security_id
open
high
low
close
volume
adjusted_close
```

---

# 8. Data Quality Requirements

Every dataset should be checked for:

- Missing values
- Duplicate records
- Invalid dates
- Incorrect prices
- Zero/negative prices
- Abnormal volume
- Corporate actions
- Stock splits
- Bonus shares
- Dividends
- Mergers
- Demergers
- Renamed companies
- Delisted companies

Each dataset should ideally contain metadata:

```text
source
ingestion_timestamp
data_timestamp
data_version
quality_flag
```

---

# 9. Exploratory Data Analysis

For each stock calculate:

## Returns

```text
Daily return
Weekly return
Monthly return
5-day return
20-day return
60-day return
```

Formula:

```text
Return_t = (Close_t - Close_(t-1)) / Close_(t-1)
```

or:

```text
Return_t = Close_t / Close_(t-1) - 1
```

## Volatility

Calculate:

```text
10-day rolling volatility
20-day rolling volatility
30-day rolling volatility
60-day rolling volatility
```

## Risk

Calculate:

```text
Maximum drawdown
Rolling drawdown
Beta
Value at Risk
Expected Shortfall
```

## Correlation

Compare each stock with:

```text
NIFTY 50
NIFTY sector index
Other stocks
```

---

# 10. Technical Analysis Engine

Initial indicators:

## Trend

```text
SMA 20
SMA 50
SMA 100
SMA 200
EMA 20
EMA 50
```

## Momentum

```text
RSI
MACD
ROC
Stochastic Oscillator
ADX
```

## Volatility

```text
ATR
Bollinger Bands
Historical Volatility
Bollinger Band Width
```

## Volume

```text
Volume Moving Average
Volume Ratio
OBV
VWAP
Delivery Ratio
```

## Relative Strength

```text
Stock Return - NIFTY Return
Stock Beta
Sector Relative Strength
```

Create a:

```text
Technical Score = 0–100
```

---

# 11. Fundamental Analysis Engine

Collect:

## Income Statement

```text
Revenue
EBITDA
EBIT
EBT
Net Income
EPS
```

## Balance Sheet

```text
Total Assets
Cash
Debt
Current Assets
Current Liabilities
Shareholders' Equity
```

## Cash Flow

```text
Operating Cash Flow
Capital Expenditure
Free Cash Flow
Investing Cash Flow
Financing Cash Flow
```

## Ratios

```text
P/E
P/B
P/S
EV/EBITDA
ROE
ROCE
Debt/Equity
Current Ratio
Interest Coverage
Dividend Yield
```

## Growth

```text
Revenue Growth
EBITDA Growth
EPS Growth
FCF Growth
Revenue CAGR
EPS CAGR
```

## Ownership

```text
Promoter Holding
FII Holding
DII Holding
Public Holding
Pledged Shares
```

Create:

```text
Fundamental Score = 0–100
```

---

# 12. News Intelligence Engine

Pipeline:

```text
News Sources
     |
     v
News Collection
     |
     v
Deduplication
     |
     v
Company / Entity Identification
     |
     v
Financial Sentiment
     |
     v
Event Classification
     |
     v
Event Impact Analysis
     |
     v
ML Features
```

## News fields

```text
news_id
headline
source
url
published_timestamp
company_id
article_timestamp
language
```

---

# 13. Financial Sentiment Analysis

Start with a finance-specific language model such as FinBERT.

Output:

```text
positive_probability
neutral_probability
negative_probability
```

Example:

```text
Positive = 0.71
Neutral  = 0.22
Negative = 0.07
```

Do not reduce everything to a simple positive/negative label.

---

# 14. Financial Event Classification

Classify news into events such as:

```text
Earnings
Dividend
Acquisition
Merger
Debt
Management Change
Regulatory Issue
Product Launch
Contract Win
Order Win
Legal Issue
Fraud Allegation
Expansion
CapEx
Credit Downgrade
Promoter Transaction
Insider Transaction
Government Policy
```

For each event estimate:

```text
event_type
event_magnitude
expected_horizon
sentiment
historical_impact
```

---

# 15. Event Impact Analysis

Instead of only asking:

> Is this news positive?

Study:

> How has this type of event historically affected this company's stock?

Example:

```text
Event: Earnings Beat

Average historical effect:

+1 day  = +1.2%
+5 days = +2.7%
+20 days = +4.1%
```

This becomes an important ML feature.

---

# 16. Macroeconomic Data

Potential features:

## India

```text
RBI Policy Rate
10Y Government Bond Yield
CPI
WPI
GDP
IIP
PMI
USD/INR
India VIX
FII Flows
DII Flows
```

## Global

```text
S&P 500
NASDAQ
Dow Jones
FTSE
Nikkei
Hang Seng
Brent Crude
WTI
Gold
US 10Y Yield
US Dollar Index
```

### Important

Every feature must respect its actual publication/availability timestamp.

Never allow the model to use information that was not available at the prediction time.

---

# 17. Market Regime Engine

Classify market conditions such as:

```text
Bull / Low Volatility
Bull / High Volatility
Bear / High Volatility
Sideways / Low Volatility
```

Possible features:

```text
NIFTY Return
India VIX
Market Breadth
Market Volatility
FII/DII Flows
USD/INR
Global Market Returns
```

Market regime becomes an ML feature.

---

# 18. Machine Learning Problem

Do not initially predict the exact future stock price.

Start with:

## Classification

Target:

```text
1 = next-day return > 0
0 = next-day return <= 0
```

Example:

```text
X = today's available information

Y = tomorrow's direction
```

---

# 19. ML Development Sequence

Use progressively more complex models.

### Baseline

```text
Naive prediction
```

### Model 1

```text
Logistic Regression
```

### Model 2

```text
Random Forest
```

### Model 3

```text
XGBoost
```

### Model 4

```text
LightGBM
```

### Advanced

```text
LSTM
GRU
Temporal Fusion Transformer
Transformer-based models
```

Do not start with deep learning.

---

# 20. Regression Model

Later predict:

```text
Next-day return
```

instead of direction.

Target:

```text
R(t+1)
```

Evaluate:

```text
MAE
RMSE
R²
```

---

# 21. Volatility Model

Build a separate model to estimate:

```text
Expected volatility
```

This will feed the risk engine.

---

# 22. Probabilistic Forecasting

Eventually generate:

```text
10th percentile
50th percentile
90th percentile
```

Example:

```text
Expected next-day range:

10% quantile = ₹2,510
Median       = ₹2,560
90% quantile = ₹2,610
```

This is preferable to claiming:

```text
Tomorrow's price = ₹2,560
```

---

# 23. Multimodal Feature Vector

A daily feature vector might contain:

```text
PRICE
open
high
low
close
volume

RETURNS
return_1d
return_5d
return_20d
return_60d

TECHNICAL
RSI
MACD
ADX
ATR
SMA20
SMA50
SMA200

VOLATILITY
volatility_10d
volatility_30d

VOLUME
volume_ratio
delivery_ratio

MARKET
NIFTY_return
NIFTY_volatility
India_VIX

SECTOR
sector_return
relative_strength

FUNDAMENTAL
PE
PB
ROE
ROCE
debt_equity
revenue_growth
EPS_growth

NEWS
news_count
sentiment_mean
sentiment_std
positive_news_ratio
negative_news_ratio
high_impact_event_count

MACRO
USDINR
Brent
US10Y
SP500_return

FII/DII
FII_net
DII_net
```

---

# 24. Model Ensemble

Potential architecture:

```text
Technical Model
       |
Fundamental Model
       |
Price ML Model
       |
News/Sentiment Model
       |
Macro Model
       |
       v
   Meta Model
       |
       v
Probability of Positive Return
```

Do not manually assume weights such as 25/25/25/25 initially. Learn ensemble weights using validation data.

---

# 25. Risk Engine

For every prediction calculate:

```text
Expected Return
Probability
Expected Volatility
Maximum Drawdown
Beta
Liquidity
VaR
Expected Shortfall
```

Output:

```text
Risk = Low / Medium / High
```

---

# 26. Investor Horizons

Use separate models/feature sets.

## Intraday

```text
1-minute
5-minute
15-minute

VWAP
Volume
Intraday volatility
Market movement
Index movement
```

## Swing / Short-term

```text
Daily OHLCV
Technical indicators
News
Earnings
Market regime
```

## Long-term

```text
Revenue Growth
EPS Growth
ROE
ROCE
FCF
Debt
Valuation
Industry
Management
Earnings Quality
```

Do not use the same model for every investment horizon.

---

# 27. Entry and Exit Engine

Treat entry/exit as a separate decision-support layer.

Potential inputs:

```text
Forecast
Volatility
Support
Resistance
ATR
Liquidity
Risk/Reward
Historical behavior
```

Example output:

```text
Potential Entry Zone
₹990–₹1,000

Potential Stop Zone
₹975

Potential Target Zone
₹1,025

Risk
Medium
```

These are model-generated hypothetical levels, not guaranteed prices.

---

# 28. Backtesting

Build a dedicated backtesting engine.

Store:

```text
timestamp
stock
signal
entry
stop_loss
target
exit
PnL
transaction_cost
slippage
```

Compare against:

```text
NIFTY Buy & Hold
```

---

# 29. Validation

Do not randomly split time-series data.

Use chronological validation:

```text
2019–2022
TRAIN

2023
VALIDATION

2024–2025
TEST
```

Then implement:

```text
Walk-forward validation
```

Eventually investigate:

```text
Purged Cross-Validation
Embargo
```

---

# 30. Evaluation Metrics

## Classification

```text
Accuracy
Precision
Recall
F1
ROC-AUC
Log Loss
Brier Score
```

## Regression

```text
MAE
RMSE
R²
```

## Trading

```text
Cumulative Return
Annualized Return
Sharpe Ratio
Sortino Ratio
Maximum Drawdown
Calmar Ratio
Win Rate
Profit Factor
Turnover
Transaction Costs
Slippage
```

---

# 31. Explainable AI

Use SHAP and related techniques.

For each prediction show:

```text
Top Positive Factors

NIFTY momentum       +0.14
Positive news        +0.11
RSI recovery         +0.09
Volume expansion     +0.07

Top Negative Factors

High valuation       -0.05
High volatility      -0.03
```

The system should be able to answer:

> Why did the model make this prediction?

---

# 32. LLM + RAG Layer

The LLM should NOT be the primary prediction engine.

Instead:

```text
Quantitative Models
       |
       v
Predictions + Evidence
       |
       v
RAG
       |
       +--> Company Filings
       +--> Financial Results
       +--> News
       +--> Corporate Announcements
       +--> Historical Analysis
       |
       v
LLM
       |
       v
Human-readable explanation
```

The LLM should explain quantitative outputs and retrieved evidence rather than invent predictions.

---

# 33. Source Reliability Hierarchy

## Tier 1

```text
NSE
BSE
SEBI
Company Filings
Annual Reports
Quarterly Results
RBI
Government Data
```

## Tier 2

```text
Reuters
Bloomberg
Major Financial Publications
```

## Tier 3

```text
Financial Websites
Analyst Reports
```

## Tier 4

```text
Blogs
Forums
Social Media
```

Higher-priority sources should receive greater trust in the RAG system.

---

# 34. Dashboard

Initial dashboard:

```text
+------------------------------------------------+
|        INDIAN EQUITY AI ANALYST                |
|                                                |
| Search: [ RELIANCE                       ] 🔍 |
+------------------------------------------------+

RELIANCE INDUSTRIES

₹XXXX.XX      +X.XX%

AI OUTLOOK
--------------------------------
Tomorrow       Bullish
Probability    64%
Expected       +0.9%
Confidence     72/100
Risk           Medium-High
--------------------------------

[Overview]
[Technical]
[Fundamentals]
[News]
[Sentiment]
[AI Forecast]
[Risk]
[Backtest]
[AI Analyst]
```

---

# 35. AI Analyst

Users should be able to ask:

```text
Why is Reliance bullish?

What are the major risks?

How did the latest earnings affect the stock?

What are the strongest fundamental factors?

What changed compared with last quarter?

How has similar news affected Reliance historically?
```

Every factual statement should ideally have a source.

---

# 36. Technology Stack

## Programming

```text
Python
SQL
```

## Data

```text
Pandas
Polars
NumPy
PyArrow
```

## ML

```text
Scikit-learn
XGBoost
LightGBM
PyTorch
```

## NLP

```text
Transformers
FinBERT
spaCy
Sentence Transformers
```

## Database

```text
PostgreSQL
pgvector
```

## API

```text
FastAPI
```

## Visualization

```text
Plotly
```

## Frontend

Start:

```text
Streamlit
```

Later:

```text
React / Next.js
```

## MLOps

```text
MLflow
Docker
GitHub
```

## Cloud

Potentially:

```text
Microsoft Azure
```

---

# 37. Recommended Project Folder

```text
indian-equity-ai/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── notebooks/
│   ├── 01_stock_data_exploration.ipynb
│   ├── 02_returns_and_risk.ipynb
│   ├── 03_technical_features.ipynb
│   ├── 04_fundamental_features.ipynb
│   ├── 05_baseline_ml.ipynb
│   ├── 06_xgboost.ipynb
│   ├── 07_news_sentiment.ipynb
│   ├── 08_multimodal_model.ipynb
│   └── 09_backtesting.ipynb
│
├── src/
│   ├── ingestion/
│   ├── preprocessing/
│   ├── features/
│   ├── technical/
│   ├── fundamentals/
│   ├── news/
│   ├── models/
│   ├── backtesting/
│   ├── risk/
│   └── api/
│
├── models/
│
├── tests/
│
├── app/
│
├── configs/
│
├── requirements.txt
│
├── README.md
└── .gitignore
```

---

# 38. Development Roadmap

## Phase 0 — Foundations

Learn:

```text
Python
Pandas
NumPy
SQL
Statistics
Probability
Basic finance
```

---

## Phase 1 — Market Data

Goal:

> Collect and clean 5 years of daily data for 5 stocks.

Deliverable:

```text
01_stock_data_exploration.ipynb
```

---

## Phase 2 — EDA

Calculate:

```text
Returns
Volatility
Drawdown
Correlation
Beta
```

Deliverable:

> Stock analytics report.

---

## Phase 3 — Technical Analysis

Build:

```text
RSI
MACD
SMA
EMA
ATR
Bollinger Bands
ADX
VWAP
OBV
```

Deliverable:

> Technical analysis engine.

---

## Phase 4 — Fundamental Analysis

Build:

```text
Financial statement processing
Ratios
Growth metrics
Quality scores
Valuation scores
```

Deliverable:

> Fundamental analysis engine.

---

## Phase 5 — Baseline ML

Build:

```text
Logistic Regression
Random Forest
XGBoost
LightGBM
```

Target:

```text
Next-day direction
```

Deliverable:

> Baseline prediction model.

---

## Phase 6 — News NLP

Build:

```text
News ingestion
Entity resolution
FinBERT sentiment
Event classification
Historical event impact
```

Deliverable:

> Financial news intelligence engine.

---

## Phase 7 — Multimodal ML

Combine:

```text
Price
Technical
Fundamental
News
Macro
Market Regime
```

Deliverable:

> Multimodal prediction model.

---

## Phase 8 — Backtesting

Implement:

```text
Walk-forward validation
Transaction costs
Slippage
Risk metrics
Benchmark comparison
```

Deliverable:

> Research-quality backtest.

---

## Phase 9 — Explainability

Implement:

```text
SHAP
Feature importance
Prediction explanations
Confidence
Uncertainty
```

Deliverable:

> Explainable prediction system.

---

## Phase 10 — LLM/RAG

Add:

```text
Company filings
News
Financial results
Corporate announcements
Model outputs
```

Deliverable:

> AI Research Analyst.

---

## Phase 11 — Application

Build:

```text
Streamlit
FastAPI
PostgreSQL
Plotly
```

Deliverable:

> Complete investor research dashboard.

---

# 39. First MVP

Do NOT attempt the entire system initially.

Your first MVP should contain only:

```text
5 stocks
5 years historical data
Daily OHLCV
Technical indicators
Fundamental metrics
Basic news
Sentiment
XGBoost
Next-day probability
Basic risk score
Streamlit dashboard
```

---

# 40. First 14-Day Action Plan

## Day 1

Set up:

```text
Python
VS Code
Jupyter
Git
GitHub
```

Create the repository.

---

## Day 2

Learn:

```text
Pandas DataFrame
CSV
DateTime
Filtering
Grouping
Merging
Missing values
```

---

## Day 3

Download historical data for:

```text
RELIANCE
TCS
INFY
HDFCBANK
ITC
```

---

## Day 4

Clean the data.

Check:

```text
Missing values
Duplicates
Date ordering
Incorrect values
```

---

## Day 5

Calculate:

```text
Daily return
Weekly return
Monthly return
```

---

## Day 6

Calculate:

```text
Rolling volatility
Maximum drawdown
Beta
Correlation
```

---

## Day 7

Visualize:

```text
Price
Returns
Volume
Volatility
Drawdown
```

---

## Day 8

Implement:

```text
SMA
EMA
RSI
MACD
```

---

## Day 9

Implement:

```text
Bollinger Bands
ATR
ADX
Volume indicators
```

---

## Day 10

Create your ML dataset.

Target:

```text
next_day_direction
```

---

## Day 11

Train:

```text
Logistic Regression
```

---

## Day 12

Train:

```text
Random Forest
XGBoost
```

---

## Day 13

Evaluate using chronological validation.

---

## Day 14

Write down:

```text
What worked?
What didn't?
Which features matter?
What should be added next?
```

---

# 41. First Major Milestone

You should consider Phase 1 complete when you can honestly say:

> "I can download five years of Indian stock-market data, clean it, calculate returns and risk metrics, generate technical features, construct a next-day prediction target, train a baseline model, and evaluate it without leaking future information."

After that, move to:

```text
Fundamentals
      ↓
News
      ↓
Sentiment
      ↓
Multimodal ML
      ↓
Risk
      ↓
Explainability
      ↓
LLM/RAG
      ↓
Dashboard
```

---

# 42. Important Research/Regulatory Notes

This project should initially be positioned as:

> **AI-Powered Indian Equity Research and Decision-Support Platform**

rather than an autonomous trading system.

Avoid claiming:

```text
Guaranteed returns
Guaranteed predictions
Certain buy/sell signals
```

If the system is eventually commercialized and gives personalized investment advice or operates within a regulated financial-services context, investigate applicable SEBI requirements and obtain appropriate professional regulatory/legal advice.

---

# 43. Current Project Definition

### Working title

**AI-Powered Indian Equity Intelligence Platform**

### Short name

**Indian Equity AI**

### Primary objective

> Build a multimodal AI/ML system that combines market, fundamental, news, sentiment, macroeconomic and technical information to generate probabilistic, explainable equity-market insights for Indian stocks.

### Initial target

```text
Next-day directional prediction
```

### Initial universe

```text
5 liquid NSE stocks
```

### Initial data period

```text
5 years
```

### Initial model

```text
Logistic Regression
Random Forest
XGBoost
```

### Final advanced model

```text
Multimodal Ensemble
+
Probabilistic Forecasting
+
Risk Engine
+
Explainable AI
+
LLM/RAG
```

---

# 44. Immediate Next Task

The **very first coding task** should be:

```text
Create:

notebooks/01_stock_data_exploration.ipynb

Then:

1. Download RELIANCE historical data
2. Load it into Pandas
3. Inspect the DataFrame
4. Clean it
5. Calculate daily returns
6. Calculate rolling volatility
7. Plot the price
8. Save the cleaned dataset
```

Once that works, expand from:

```text
1 stock
      ↓
5 stocks
      ↓
50 stocks
      ↓
full selected NSE universe
```

This prevents the project from becoming overwhelming and gives you a working component at every stage.

---

## Important principle

**Do not start with AI. Start with data.**

The correct progression is:

```text
DATA
  ↓
UNDERSTANDING
  ↓
FEATURES
  ↓
BASELINE
  ↓
MACHINE LEARNING
  ↓
VALIDATION
  ↓
NEWS/NLP
  ↓
MULTIMODAL MODEL
  ↓
RISK
  ↓
EXPLAINABILITY
  ↓
LLM
  ↓
APPLICATION
```

That is the roadmap this project should follow.
