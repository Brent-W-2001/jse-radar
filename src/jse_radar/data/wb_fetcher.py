"""
World Bank Fetcher — downloads SA macroeconomic indicators via wbgapi.

wbgapi is the official Python client for the World Bank Open Data API.
It's clean, requires no API key, and can return pandas DataFrames directly.

IMPORTANT: wbgapi has two similarly-named functions that behave very
differently:
  - wb.data.get()       → returns a single value/dict. NOT what we want.
  - wb.data.DataFrame()  → returns a proper 2D pandas DataFrame. This is
                           the correct function for fetching a full table
                           of indicators across years.

This was the source of a bug where the fetcher crashed with
'dict' object has no attribute 'empty' — we were calling .get()
which returns a dict-like object, not a DataFrame, so calling
.empty on it failed.

World Bank data is annual — it gives us structural context
(debt/GDP, current account balance, long-run growth) that FRED's
series complement with higher-frequency monthly data.

Output file: data/raw/worldbank/worldbank_YYYYMMDD.parquet
Column structure: year | gdp_growth_pct | inflation_cpi_pct | ...
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import pandas as pd

from jse_radar.config import WB_INDICATORS, WB_COUNTRY_CODE, RAW_WB_DIR
from jse_radar.data.fetcher import DataFetcher
from jse_radar.utils.logger import get_logger

logger = get_logger(__name__)


class WorldBankFetcher(DataFetcher):
    """Fetches South African macroeconomic indicators from the World Bank API."""

    def __init__(
        self,
        indicators: dict[str, str] | None = None,
        country: str = WB_COUNTRY_CODE,
        raw_dir: Path = RAW_WB_DIR,
    ) -> None:
        super().__init__(raw_dir)
        self.indicators = indicators or WB_INDICATORS
        self.country = country

    def fetch(
        self,
        start_date: str = "2000-01-01",
        end_date: str | None = None,
    ) -> None:
        """
        Fetch all World Bank indicators for South Africa.

        wb.data.DataFrame() returns a DataFrame with indicators as the
        index and years as columns (e.g. 'YR2020') by default. We pass
        numericTimeKeys=True so years come back as plain integers, and
        columns='series' so indicators become columns instead — this
        gives us directly the shape we want: year as rows, indicator as
        columns, with almost no reshaping needed.
        """
        try:
            import wbgapi as wb
        except ImportError:
            raise ImportError(
                "wbgapi is not installed. Run: pip install wbgapi"
            )

        start_year = int(start_date[:4])
        end_year   = int(end_date[:4]) if end_date else datetime.now().year

        logger.info(
            f"Fetching {len(self.indicators)} World Bank indicators "
            f"for {self.country}: {start_year} → {end_year}"
        )

        indicator_ids = list(self.indicators.keys())

        try:
            # economy=self.country must be a list for consistent shape
            # columns='series' pivots indicators into columns directly
            # numericTimeKeys=True gives us plain ints instead of 'YR2020'
            raw = wb.data.DataFrame(
                indicator_ids,
                economy=self.country,
                time=range(start_year, end_year + 1),
                columns="series",
                numericTimeKeys=True,
                skipBlanks=False,
            )
        except Exception as e:
            logger.error(f"World Bank API error: {e}")
            raise

        if raw is None or raw.empty:
            logger.warning("World Bank returned empty DataFrame.")
            return

        # ── Reshape ────────────────────────────────────────────────────────────
        # With a single economy and columns='series', the index is the
        # time dimension (years) and columns are indicator codes.
        # We reset the index so 'year' becomes a normal column.
        df = raw.reset_index()

        # The index column is named 'time' or sometimes 'index' depending
        # on wbgapi version — normalise it to 'year'.
        index_col = df.columns[0]
        df = df.rename(columns={index_col: "year"})

        # Rename indicator codes to our human-readable labels
        df = df.rename(columns=self.indicators)

        # Ensure year is a clean integer and sort chronologically
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        df = df.dropna(subset=["year"]).sort_values("year").reset_index(drop=True)

        filename = f"worldbank_{datetime.now().strftime('%Y%m%d')}.parquet"
        output_path = self.raw_dir / filename
        df.to_parquet(output_path, index=False, engine="pyarrow")

        logger.info(
            f"Saved World Bank data: {len(df):,} rows × {len(self.indicators)} "
            f"indicators to {output_path}"
        )

    def load(self) -> pd.DataFrame:
        """Load the most recently fetched World Bank parquet file."""
        path = self._latest_file("worldbank_*.parquet")
        if path is None:
            raise FileNotFoundError(
                f"No World Bank data in {self.raw_dir}. Run fetch() first."
            )
        logger.info(f"Loading World Bank data from {path}")
        return pd.read_parquet(path, engine="pyarrow")