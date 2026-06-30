# Changelog

## 2026-06-30 — Modeling layer added

### Markov chain
- `src/jse_radar/modeling/markov_chain.py`: first-order Markov chain
  estimating regime transition probabilities from observed history,
  with confidence flagging for transition rows backed by too few
  historical occurrences, and `expected_regime_duration()` via the
  geometric-distribution formula
- Cross-validated against the regime notebook's own hand-counted spell
  lengths: exact agreement for HIKING_HIGH_INFLATION (6.8 months via
  both independent methods)
- 16 tests, 76% coverage

### Regime predictor
- `src/jse_radar/modeling/regime_predictor.py`: logistic regression
  predicting P(stock outperforms ALSI over next 21 trading days |
  momentum/RSI/trend signals, composite regime). Strict chronological
  train/test splitting (no random shuffling — the modeling-layer
  equivalent of the master builder's no-look-ahead-bias guarantee).
  Handles unseen/missing regimes between train and test via a fixed,
  reindexed one-hot column set
- 13 tests, 100% coverage — every line of production code directly tested

### Walk-forward backtester
- `src/jse_radar/modeling/backtester.py`: monthly walk-forward
  evaluation — refits the model at every rebalance date using only
  prior data, selects top-5 predicted picks, measures realised return
  vs the ALSI, applies a 15bps rebalancing cost haircut
- 10 tests, 91% coverage, including an explicit re-derivation check
  proving no period's training data ever includes its own test date

### Honest result
- **111 monthly rebalance periods evaluated. Hit rate: 49.5%. Average
  net excess return: +0.30%/month. Sharpe-style ratio: 0.21.**
- The model does NOT demonstrate a reliable edge over holding the ALSI
  — statistically indistinguishable from chance. Documented in
  `notebooks/03_modeling_and_backtest.ipynb` as a reproducible baseline,
  not treated as a failure: any future modeling attempt on this data
  has a clear, honestly-measured bar to beat.

### Testing
- Full suite now at 108 tests across 10 files (up from 77 across 7)

---

## 2026-06-29 — Project complete

### Data pipeline

- Equity fetcher: 28 JSE tickers via yfinance (`.JO` suffix), daily OHLCV
  back to 2015, removed 3 delisted tickers (AMS.JO, BHP.JO, RMI.JO)
- Macro fetcher: 8 South African indicators via FRED API (T-bill rate,
  10y government bond yield, CPI all/core, ZAR/USD, USD/EUR, exports,
  imports), corrected after 5 retired/incorrect FRED series IDs were
  identified and replaced
- World Bank fetcher: 7 structural indicators (GDP growth, inflation,
  unemployment, current account/GDP, government debt/GDP, exports/GDP,
  imports/GDP) — fixed a bug where `wb.data.get()` (returns a dict) was
  being called instead of `wb.data.DataFrame()` (returns a proper
  DataFrame), which had silently broken this fetcher entirely

### Processing layer

- Equity processor: forward-fill with a 5-day cap, daily and log
  returns, 21-day annualised volatility, 52-week high/low, calendar features
- Macro processor: mixed-frequency resampling (FX columns averaged,
  level/rate columns last-value), 3-month forward-fill cap, CPI
  year-on-year, real interest rate, ZAR/USD month-on-month change
- Master builder: asof merge joining equity + FRED macro + World Bank
  data into one frame, explicitly verified to have no look-ahead bias;
  fixed a datetime precision mismatch (ms vs us) that initially broke
  the merge

### Analysis layer

- Signal engine: momentum (1/3/6/12 month), cross-sectional z-scores,
  RSI-14, MA50/MA200 trend signal, mean reversion z-score, composite
  momentum score — fixed an RSI edge case where a stock with zero
  losing days in its window returned NaN forever instead of the
  correct RSI = 100
- Macro regime classifier: rate regime (hiking/cutting vs 3-month
  rolling average), inflation regime (vs SARB 3-6% target band),
  currency regime, composite regime label, regime duration and
  independent spell counting
- Correlation analyser: 90-day rolling correlations between equity
  returns and macro variables, static return correlation matrix

### Notebooks

- `01_eda_jse_equities.ipynb`: data quality check, price history,
  return distributions, macro overview, ZAR/USD correlation by ticker
- `02_regime_analysis.ipynb`: regime duration and spell-count analysis,
  forward-return heatmap by ticker × regime (with minimum observation
  AND minimum spell-count filtering), momentum signal efficacy by
  regime, hiking/cutting hedge analysis, current-regime stock ranking.
  Identified that only 3 of 6 macro regimes (HIKING_HIGH_INFLATION,
  HIKING_TARGET_INFLATION, CUTTING_TARGET_INFLATION) have sufficient
  independent historical spells for reliable conclusions — an initially
  promising HIKING_LOW_INFLATION signal was correctly discarded after
  the spell-count filter showed it was based on a single 22-day episode

### Dashboard

- Four-tab Plotly Dash app: Market Overview (candlestick + MA50/MA200
  - volume + RSI), Macro Environment (FRED indicators + regime banner
  - regime history timeline), Signals (momentum bar chart + sortable
    table), Correlations (rolling correlation chart + static heatmap)
- Runs on `http://127.0.0.1:8050` via `python -m jse_radar.dashboard.app`

### Automation

- `scripts/run_pipeline.py`: full pipeline entry point — fetch, process,
  build, analyse, quality-check, in one command
- `scripts/refresh.bat` + Windows Task Scheduler: automated weekday
  07:00 refresh
- `scripts/verify_refresh.py`: on-demand data freshness report

### Data quality

- `src/jse_radar/utils/data_quality.py`: post-pipeline checks for
  missing trading days, frozen/stale price feeds, macro series
  completeness, and master frame row-count regression versus the
  previous run — logs warnings/errors without blocking the pipeline

### Testing

- 77 tests across 7 files: `test_config`, `test_master_builder`,
  `test_signals`, `test_macro_regime`, `test_correlation`,
  `test_equity_processor`, `test_macro_processor`, `test_data_quality`
- Proves: no look-ahead bias in the asof merge, correct momentum/RSI/
  trend-signal mathematics, correct regime classification and spell
  counting, rolling correlation window correctness and bounds, correct
  forward-fill/return/volatility calculations, and that the data
  quality checks fire exactly when they should (and stay silent when
  they shouldn't)

---

## 2026-06-12 — First successful end-to-end pipeline run

- 79,513 rows × 28 JSE tickers fetched via yfinance (2015–2026)
- 138 monthly macro rows fetched via FRED API
- Master frame: 79,513 rows × 29 columns built successfully
