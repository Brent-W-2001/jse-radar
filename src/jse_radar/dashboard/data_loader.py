"""
Dashboard data loader — loads and caches all parquet files at startup.

Why load everything at startup?
  Dash runs as a web server. If we loaded parquet files inside callbacks,
  every user interaction would trigger a file read. Loading once at startup
  and keeping DataFrames in memory means interactions are instant.

  For a personal analytics tool with datasets this size (~80k rows),
  in-memory is perfectly appropriate. A production multi-user app would
  use a database or Redis cache instead.
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Optional

from jse_radar.config import PROC_MASTER_DIR, PROC_MACRO_DIR
from jse_radar.utils.logger import get_logger

logger = get_logger(__name__)


def _latest(directory: Path, pattern: str) -> Optional[Path]:
    """Return the most recently modified file matching pattern, or None."""
    files = sorted(directory.glob(pattern), key=lambda f: f.stat().st_mtime)
    return files[-1] if files else None


def load_master() -> pd.DataFrame:
    """Load the most enriched master frame available."""
    for pattern in [
        "master_regimes_*.parquet",
        "master_signals_*.parquet",
        "master_*.parquet",
    ]:
        path = _latest(PROC_MASTER_DIR, pattern)
        if path:
            logger.info(f"Dashboard loading master from: {path.name}")
            df = pd.read_parquet(path, engine="pyarrow")
            df["date"] = pd.to_datetime(df["date"])
            return df
    raise FileNotFoundError("No master parquet found. Run the pipeline first.")


def load_macro() -> pd.DataFrame:
    """Load processed macro data."""
    path = _latest(PROC_MACRO_DIR, "macro_processed_*.parquet")
    if not path:
        raise FileNotFoundError("No macro parquet found. Run the pipeline first.")
    logger.info(f"Dashboard loading macro from: {path.name}")
    df = pd.read_parquet(path, engine="pyarrow")
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_regimes() -> Optional[pd.DataFrame]:
    """Load macro regime classifications if available."""
    path = _latest(PROC_MACRO_DIR, "macro_regimes_*.parquet")
    if not path:
        logger.warning("No regime parquet found — regime tab will be limited.")
        return None
    df = pd.read_parquet(path, engine="pyarrow")
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_correlations() -> Optional[pd.DataFrame]:
    """Load rolling correlations if available."""
    path = _latest(PROC_MASTER_DIR, "rolling_correlations_*.parquet")
    if not path:
        logger.warning("No correlations parquet found.")
        return None
    df = pd.read_parquet(path, engine="pyarrow")
    df["date"] = pd.to_datetime(df["date"])
    return df


class DashboardData:
    """
    Container that loads and holds all dashboard datasets.
    Instantiated once at app startup.

    Attributes are None if the underlying file doesn't exist yet —
    the app handles this gracefully rather than crashing.
    """

    def __init__(self) -> None:
        logger.info("Loading dashboard data...")
        self.master       = load_master()
        self.macro        = load_macro()
        self.regimes      = load_regimes()
        self.correlations = load_correlations()

        # Derived convenience attributes
        self.tickers      = sorted(self.master["ticker"].unique())
        self.ticker_names = (
            self.master.drop_duplicates("ticker")
            .set_index("ticker")["name"]
            .to_dict()
            if "name" in self.master.columns else {}
        )
        self.macro_vars   = [
            c for c in [
                "zar_usd_mom_pct", "tbill_rate", "cpi_yoy_pct",
                "real_tbill_rate", "exports_value", "imports_value",
            ]
            if c in self.master.columns
        ]
        self.date_min = self.master["date"].min()
        self.date_max = self.master["date"].max()

        logger.info(
            f"Dashboard data ready. "
            f"{len(self.tickers)} tickers, "
            f"{self.date_min.date()} → {self.date_max.date()}"
        )