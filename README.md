# JSE Radar

> A macroeconomic data pipeline, analysis suite, and interactive dashboard
> for the Johannesburg Stock Exchange (JSE) and South African macro
> indicators. Scans the market landscape so you don't have to.

## What this project does

- Fetches daily JSE equity prices (28 tickers) via yfinance
- Fetches South African macro indicators (interest rates, inflation,
  exchange rates, trade) via the FRED API
- Fetches structural macro data (GDP growth, government debt, current
  account) via the World Bank API
- Computes momentum, RSI, moving-average trend signals, and macro regime
  classification (hiking/cutting × inflation level)
- Joins everything into one analytical frame with no look-ahead bias
- Serves it all through a four-tab interactive Plotly Dash dashboard
- Refreshes automatically on a schedule via Windows Task Scheduler
- Runs automated data quality checks after every pipeline run
- Models regime transition probabilities (Markov chain) and tests
  whether signals + regime can predict stock outperformance (logistic
  regression), evaluated with a proper walk-forward backtest
- Is backed by 108 tests covering every core calculation in the pipeline
  and modeling layer

## Project structure

```
jse-radar/
├── .github/
│   └── workflows/
│       └── ci.yml                       # GitHub Actions CI — runs pytest on push/PR
├── scripts/
│   ├── run_pipeline.py                   # Full pipeline entry point (fetch → process → analyse → quality-check)
│   ├── run_backtest.py                   # Walk-forward backtest entry point
│   ├── check_markov.py                   # Inspect fitted Markov chain vs regime history
│   ├── refresh.bat                       # Scheduled refresh script (Windows Task Scheduler)
│   └── verify_refresh.py                 # Data freshness checker — run any time
├── src/jse_radar/
│   ├── __init__.py
│   ├── config.py                         # Paths, ticker universe, FRED/WB series, constants
│   ├── data/
│   │   ├── __init__.py
│   │   ├── fetcher.py                    # Abstract base class for all fetchers
│   │   ├── equity_fetcher.py             # JSE prices via yfinance (.JO tickers)
│   │   ├── macro_fetcher.py              # SA macro via FRED API
│   │   ├── wb_fetcher.py                 # SA structural indicators via World Bank wbgapi
│   │   └── pipeline.py                   # Orchestrates all three fetchers
│   ├── analysis/
│   │   ├── __init__.py                   # Import shortcuts for all three modules
│   │   ├── signals.py                    # Momentum, RSI, MA crossover, mean reversion
│   │   ├── macro_regime.py               # Rate/inflation/currency regime classification
│   │   └── correlation.py                # 90-day rolling correlations, static matrix
│   ├── processing/
│   │   ├── __init__.py
│   │   ├── equity_processor.py           # Clean equity data, compute returns/volatility
│   │   ├── macro_processor.py            # Resample, forward-fill, derive macro indicators
│   │   └── master_builder.py             # Asof-merge equity + macro + World Bank into one frame
│   ├── modeling/
│   │   ├── __init__.py
│   │   ├── markov_chain.py               # Regime transition probabilities, expected duration
│   │   ├── regime_predictor.py           # Logistic regression: P(outperform ALSI | signals, regime)
│   │   └── backtester.py                 # Walk-forward evaluation vs buy-and-hold ALSI
│   ├── dashboard/
│   │   ├── __init__.py
│   │   ├── app.py                        # Dash application factory — runs on :8050
│   │   ├── data_loader.py                # Loads all parquet files at startup
│   │   ├── layout.py                     # Four-tab layout, dark theme, colour palette
│   │   └── callbacks.py                  # All interactive chart/table behaviour
│   └── utils/
│       ├── __init__.py
│       ├── logger.py                     # Rotating file + console logger
│       └── data_quality.py               # Post-pipeline checks: missing data, frozen feeds,
│                                          #   macro completeness, row-count regression
├── notebooks/
│   ├── 01_eda_jse_equities.ipynb         # EDA: price history, distributions, correlations
│   ├── 02_regime_analysis.ipynb          # Forward returns by regime, momentum efficacy,
│   │                                     #   spell-count validation, current-regime stock ranking
│   └── 03_modeling_and_backtest.ipynb    # Walk-forward backtest results, cumulative excess
│                                          #   return, coefficient inspection, honest conclusion
├── tests/
│   ├── __init__.py
│   ├── test_config.py                    # Smoke tests for config paths
│   ├── test_master_builder.py            # Asof merge correctness, no look-ahead bias
│   ├── test_signals.py                   # Momentum, RSI, trend signal, z-score maths
│   ├── test_macro_regime.py              # Regime classification, spell counting
│   ├── test_correlation.py               # Rolling correlation window correctness
│   ├── test_equity_processor.py          # Forward-fill, returns, volatility, 52w high/low
│   ├── test_macro_processor.py           # Resampling, CPI YoY, real rate, ZAR/USD MoM
│   ├── test_data_quality.py              # Each quality check fires correctly, and stays silent
│   │                                     #   on healthy data
│   ├── test_markov_chain.py              # Transition probabilities, expected duration,
│   │                                     #   confidence flagging
│   ├── test_regime_predictor.py          # Outperformance labelling, chronological split,
│   │                                     #   regime encoding consistency
│   └── test_backtester.py                # Walk-forward leakage guarantee, cost arithmetic,
│                                          #   summary metrics
├── data/                                 # gitignored — never committed
│   ├── raw/
│   │   ├── equities/                     # equities_YYYYMMDD.parquet
│   │   ├── macro/                        # macro_YYYYMMDD.parquet
│   │   └── worldbank/                    # worldbank_YYYYMMDD.parquet
│   ├── processed/
│   │   ├── equities/                     # equities_processed_YYYYMMDD.parquet
│   │   ├── macro/                        # macro_processed_YYYYMMDD.parquet
│   │   │                                 # macro_regimes_YYYYMMDD.parquet
│   │   └── master/                       # master_YYYYMMDD.parquet
│   │                                     # master_signals_YYYYMMDD.parquet
│   │                                     # master_regimes_YYYYMMDD.parquet
│   │                                     # rolling_correlations_YYYYMMDD.parquet
│   │                                     # markov_transition_matrix_YYYYMMDD.parquet
│   └── external/
├── dashboard/                            # Standalone dashboard exports/static assets (empty)
├── infra/                                # Infrastructure as code (reserved, unused)
├── reports/
│   └── figures/                          # Output charts and reports
├── logs/                                 # gitignored
│   ├── jse_radar.log                     # Rotating pipeline log
│   └── scheduler.log                     # Scheduled refresh run history
├── .env                                  # API keys — gitignored, never committed
├── .env.example                          # Template showing required env vars
├── .gitignore
├── CHANGELOG.md
├── environment.yml
├── pyproject.toml
└── README.md
```

## Quickstart

```bash
# Clone and create the conda environment
git clone https://github.com/Brent-W-2001/jse-radar.git
cd jse-radar
conda env create -f environment.yml
conda activate jse-radar
pip install -e ".[dev]"

# Add your FRED API key
# Create a .env file in the repo root (see .env.example):
#   FRED_API_KEY=your_key_here
#   LOG_LEVEL=INFO

# Run the full pipeline (fetch -> process -> analyse -> quality-check)
python scripts/run_pipeline.py --start 2015-01-01

# Run the walk-forward backtest
python scripts/run_backtest.py

# Run tests
pytest

# Check data freshness any time
python scripts/verify_refresh.py

# Launch the dashboard
python -m jse_radar.dashboard.app
# then open http://127.0.0.1:8050

# Launch JupyterLab for the analysis and modeling notebooks
jupyter lab
```

## Scheduled refresh

`scripts/refresh.bat` is configured to run via Windows Task Scheduler on
weekdays at 07:00, keeping the dashboard's underlying data current
without manual intervention. Run `python scripts/verify_refresh.py`
any time to confirm how fresh the data currently is. Every run also
passes through `src/jse_radar/utils/data_quality.py`, which checks for
missing trading days, frozen price feeds, macro series completeness,
and unexpected row-count drops versus the previous run.

## Modeling

`notebooks/03_modeling_and_backtest.ipynb` documents a walk-forward
backtest of a logistic regression model (momentum/RSI/trend signals,
conditioned on macro regime) predicting whether a stock will outperform
the ALSI over the next month. **The honest result is negative**: 111
monthly rebalance periods, a 49.5% hit rate, and a Sharpe-style ratio of
0.21 — statistically indistinguishable from chance, even after a proper
chronological train/test split and realistic rebalancing costs. This is
documented as a reproducible baseline for any future modeling work, not
treated as a failure to fix.

A separate `RegimeMarkovChain` model estimates regime transition
probabilities directly from the regime history and was cross-validated
against the regime notebook's own hand-counted spell-length averages —
the two independently-derived numbers agreed exactly for
`HIKING_HIGH_INFLATION` (6.8 months both ways).

## Data

Raw and processed data are gitignored and never committed to the
repository. Run `scripts/run_pipeline.py` to fetch and build everything
locally — it pulls equities (yfinance), macro indicators (FRED), and
structural indicators (World Bank), then runs the full analysis layer
and a final data quality check.

## Testing

```bash
pytest -v
```

108 tests cover every core calculation in the pipeline and modeling
layer: the asof merge (proving no look-ahead bias), momentum/RSI/trend
signal maths, macro regime classification and spell counting, rolling
correlation window correctness, the equity/macro processors' resampling
and derived indicators, the data quality checks (firing correctly and
staying silent on healthy data), the Markov chain's transition
probability estimation, and the backtester's walk-forward leakage
guarantee.

## Author

Brent Williams — brentw2001@gmail.com

## License

MIT
