# jse-radar

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
- Is backed by 62 tests covering every core calculation in the pipeline

## Project structure

```
jse-radar/
├── .github/
│   └── workflows/
│       └── ci.yml                       # GitHub Actions CI — runs pytest on push/PR
├── scripts/
│   ├── run_pipeline.py                   # Full pipeline entry point (fetch → process → analyse)
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
│   ├── dashboard/
│   │   ├── __init__.py
│   │   ├── app.py                        # Dash application factory — runs on :8050
│   │   ├── data_loader.py                # Loads all parquet files at startup
│   │   ├── layout.py                     # Four-tab layout, dark theme, colour palette
│   │   └── callbacks.py                  # All interactive chart/table behaviour
│   └── utils/
│       ├── __init__.py
│       └── logger.py                     # Rotating file + console logger
├── notebooks/
│   ├── 01_eda_jse_equities.ipynb         # EDA: price history, distributions, correlations
│   └── 02_regime_analysis.ipynb          # Forward returns by regime, momentum efficacy,
│                                          #   spell-count validation, current-regime stock ranking
├── tests/
│   ├── __init__.py
│   ├── test_config.py                    # Smoke tests for config paths
│   ├── test_master_builder.py            # Asof merge correctness, no look-ahead bias
│   ├── test_signals.py                   # Momentum, RSI, trend signal, z-score maths
│   ├── test_macro_regime.py              # Regime classification, spell counting
│   ├── test_correlation.py               # Rolling correlation window correctness
│   ├── test_equity_processor.py          # Forward-fill, returns, volatility, 52w high/low
│   └── test_macro_processor.py           # Resampling, CPI YoY, real rate, ZAR/USD MoM
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
│   └── external/
├── dashboard/                            # Standalone dashboard exports/static assets (empty)
├── infra/                                # Infrastructure as code (reserved for future use)
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

# Run the full pipeline (fetch -> process -> analyse)
python scripts/run_pipeline.py --start 2015-01-01

# Run tests
pytest

# Check data freshness any time
python scripts/verify_refresh.py

# Launch the dashboard
python -m jse_radar.dashboard.app
# then open http://127.0.0.1:8050

# Launch JupyterLab for the analysis notebooks
jupyter lab
```

## Scheduled refresh

`scripts/refresh.bat` is configured to run via Windows Task Scheduler on
weekdays at 07:00, keeping the dashboard's underlying data current
without manual intervention. Run `python scripts/verify_refresh.py`
any time to confirm how fresh the data currently is.

## Data

Raw and processed data are gitignored and never committed to the
repository. Run `scripts/run_pipeline.py` to fetch and build everything
locally — it pulls equities (yfinance), macro indicators (FRED), and
structural indicators (World Bank), then runs the full analysis layer.

## Testing

```bash
pytest -v
```

62 tests cover every core calculation in the pipeline: the asof merge
(proving no look-ahead bias), momentum/RSI/trend signal maths, macro
regime classification and spell counting, rolling correlation window
correctness, and the equity/macro processors' resampling and derived
indicators.

## Author

Brent Williams — brentw2001@gmail.com

## License

MIT
