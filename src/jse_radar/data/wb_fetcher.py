"""
World Bank Fetcher — downloads SA macroeconomic indicators via wbgapi.

wbgapi is the official Python client for the World Bank Open Data API.
It's clean, requires no API key, and returns pandas DataFrames directly.

World Bank data is annual or quarterly — it gives us structural context
(debt/GDP, current account balance, long-run growth) that FRED's series
complement with higher-frequency monthly data.

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

        wbgapi.data.get() returns a DataFrame indexed by (economy, time).
        We select South Africa only, pivot indicators into columns,
        and convert the "YR2020" style year strings to integers.
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

        indicator_ids   = list(self.indicators.keys())
        indicator_names = list(self.indicators.values())

        try:
            # wb.data.get returns a DataFrame with MultiIndex (economy, series, time)
            # mrv=None means fetch all available years (not just most recent)
            raw = wb.data.get(
                indicator_ids,
                economy=self.country,
                time=range(start_year, end_year + 1),
            )
        except Exception as e:
            logger.error(f"World Bank API error: {e}")
            raise

        if raw.empty:
            logger.warning("World Bank returned empty DataFrame.")
            return

        # The DataFrame index has levels: economy, series, time
        # We only have one economy (ZAF) so we drop that level.
        # Then we unstack the series level to get one column per indicator.
        df = raw.reset_index()

        # Rename the time column and extract year as integer
        # World Bank time format is "YR2020", "YR2021", etc.
        if "time" in df.columns:
            df["year"] = df["time"].str.replace("YR", "").astype(int)
            df = df.drop(columns=["time"])

        # Drop the economy column (all ZAF)
        if "economy" in df.columns:
            df = df.drop(columns=["economy"])

        # Pivot: one column per indicator
        if "series" in df.columns and "value" in df.columns:
            df = df.pivot(index="year", columns="series", values="value")
            df.columns.name = None
            df = df.rename(columns=self.indicators)
            df = df.reset_index()

        df = df.sort_values("year").reset_index(drop=True)

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