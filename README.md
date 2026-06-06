# jse-radar

> A macroeconomic data pipeline, analysis suite, and dashboard for the
> Johannesburg Stock Exchange (JSE) and South African macro indicators.
> Scans the market landscape so you don't have to.

## Project structure

```
jse-radar/
├── .github/workflows/    # CI/CD — GitHub Actions
├── src/jse_radar/        # Python package (importable as `jse_radar`)
│   ├── config.py         # Paths and environment config
│   ├── data/             # Fetchers, loaders, validators
│   ├── analysis/         # Signals, indicators, models
│   └── dashboard/        # Plotly Dash app
├── notebooks/            # Exploratory analysis
├── tests/                # pytest test suite
├── data/                 # gitignored — never committed
│   ├── raw/              # Downloaded source data
│   ├── processed/        # Cleaned, feature-engineered data
│   └── external/         # Reference data (indices, calendars)
├── dashboard/            # Standalone dashboard assets
├── infra/                # Infrastructure as code
├── reports/figures/      # Output charts and reports
├── environment.yml       # Conda environment definition
└── pyproject.toml        # Package config, tool settings
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
