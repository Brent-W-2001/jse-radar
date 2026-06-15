# JSE-RADAR  

> A macroeconomic data pipeline, analysis suite, and dashboard for the
> Johannesburg Stock Exchange (JSE) and South African macro indicators.
> Scans the market landscape so you don't have to.

## Project structure

```
jse-radar/
├── .github/
│   └── workflows/
│       └── ci.yml                    # GitHub Actions CI
├── scripts/
│   └── run_pipeline.py               # Full pipeline entry point (fetch → process → analyse)
├── src/jse_radar/
│   ├── __init__.py
│   ├── config.py                     # Paths, ticker universe, FRED/WB series, constants
│   ├── data/
│   │   ├── __init__.py
│   │   ├── fetcher.py                # Abstract base class for all fetchers
│   │   ├── equity_fetcher.py         # JSE prices via yfinance (.JO tickers)
│   │   ├── macro_fetcher.py          # SA macro via FRED API
│   │   ├── wb_fetcher.py             # SA indicators via World Bank wbgapi
│   │   └── pipeline.py               # Orchestrates all three fetchers
│   ├── analysis/
│   │   ├── __init__.py               # Import shortcuts for all three modules
│   │   ├── signals.py                # Momentum, RSI, MA crossover, mean reversion
│   │   ├── macro_regime.py           # Rate/inflation/currency regime classification
│   │   └── correlation.py            # 90-day rolling correlations, static matrix
│   ├── processing/
│   │   ├── __init__.py
│   │   ├── equity_processor.py       # Clean equity data, compute returns/volatility
│   │   ├── macro_processor.py        # Resample, forward-fill, derive macro indicators
│   │   └── master_builder.py         # Asof-merge equity + macro into one frame
│   ├── dashboard/
│   │   └── __init__.py               # Plotly Dash app (to be built)
│   └── utils/
│       ├── __init__.py
│       └── logger.py                 # Rotating file + console logger
├── notebooks/
│   └── 01_eda_jse_equities.ipynb     # EDA: price history, distributions, correlations
├── tests/
│   ├── __init__.py
│   └── test_config.py                # Smoke tests for config paths
├── data/                             # gitignored — never committed
│   ├── raw/
│   │   ├── equities/                 # equities_YYYYMMDD.parquet
│   │   ├── macro/                    # macro_YYYYMMDD.parquet
│   │   └── worldbank/                # worldbank_YYYYMMDD.parquet
│   ├── processed/
│   │   ├── equities/                 # equities_processed_YYYYMMDD.parquet
│   │   ├── macro/                    # macro_processed_YYYYMMDD.parquet
│   │   └── master/                   # master_YYYYMMDD.parquet
│   │                                 # master_signals_YYYYMMDD.parquet
│   │                                 # master_regimes_YYYYMMDD.parquet
│   │                                 # rolling_correlations_YYYYMMDD.parquet
│   └── external/
├── dashboard/                        # Standalone dashboard assets (to be built)
├── infra/                            # Infrastructure as code (to be built)
├── reports/
│   └── figures/                      # Output charts and reports
├── logs/
│   └── jse_radar.log                 # Rotating log file (gitignored)
├── .env                              # API keys — gitignored, never committed
├── .env.example                      # Template showing required env vars
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

# Run tests
pytest

# Launch JupyterLab
jupyter lab
```

## Data

Raw and processed data are gitignored and never committed to the repository.
Run the pipeline scripts in `src/jse_radar/data/` to fetch data locally.

## Author

Brent Williams — brentw2001@gmail.com

## License

MIT
