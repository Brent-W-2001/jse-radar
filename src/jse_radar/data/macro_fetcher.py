"""
FRED Macro Fetcher — downloads SA macroeconomic series via the FRED API.

FRED (Federal Reserve Economic Data) is maintained by the St. Louis Fed
and hosts hundreds of South African time series sourced from the SARB,
Stats SA, and the IMF. It's free, reliable, and has a clean REST API.

You need a free API key from: https://fred.stlouisfed.org/docs/api/api_key.html
Put it in your .env file as: FRED_API_KEY=your_key_here

How the FRED API works:
  GET https://api.stlouisfed.org/fred/series/observations
  Parameters: series_id, observation_start, observation_end, api_key, file_type
  Returns: JSON with a list of {date, value} observations

We loop through each series in FRED_SERIES and combine them into
a single wide DataFrame (one column per indicator, one row per date).

Output file: data/raw/macro/macro_YYYYMMDD.parquet
Column structure: date | repo_rate | cpi_all | zar_usd | ... (one col per series)
"""

from __future__ import annotations

from datetime import datetime, date
from pathlib import Path
import time

import httpx
import pandas as pd

from jse_radar.config import FRED_API_KEY, FRED_SERIES, RAW_MACRO_DIR, DEFAULT_START_DATE
from jse_radar.data.fetcher import DataFetcher
from jse_radar.utils.logger import get_logger

logger = get_logger(__name__)

FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"


class MacroFetcher(DataFetcher):
    """Fetches SA macroeconomic series from the FRED API."""

    def __init__(
        self,
        series: dict[str, str] | None = None,
        raw_dir: Path = RAW_MACRO_DIR,
        api_key: str | None = None,
    ) -> None:
        super().__init__(raw_dir)
        self.series = series or FRED_SERIES
        self.api_key = api_key or FRED_API_KEY

        if not self.api_key:
            raise ValueError(
                "FRED_API_KEY not set. Add it to your .env file. "
                "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
            )

    def _fetch_one_series(
        self,
        series_id: str,
        label: str,
        start_date: str,
        end_date: str,
        client: httpx.Client,
    ) -> pd.Series:
        """
        Fetch a single FRED series and return it as a pandas Series
        with a DatetimeIndex.

        We use httpx (a modern, sync/async HTTP client) rather than
        requests because it has better timeout handling and is what
        we'll use if we go async later.

        FRED returns "." for missing values — we convert those to NaN.
        """
        params = {
            "series_id":         series_id,
            "observation_start": start_date,
            "observation_end":   end_date,
            "api_key":           self.api_key,
            "file_type":         "json",
        }

        try:
            resp = client.get(FRED_BASE_URL, params=params, timeout=30.0)
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching {series_id}: {e.response.status_code}")
            raise
        except httpx.TimeoutException:
            logger.error(f"Timeout fetching {series_id}")
            raise

        data = resp.json()

        if "observations" not in data:
            logger.warning(f"No observations key in FRED response for {series_id}")
            return pd.Series(name=label, dtype=float)

        observations = data["observations"]

        # Build a Series: index = date, values = float
        # FRED uses "." for missing values, which we replace with NaN
        s = pd.Series(
            data={
                obs["date"]: float(obs["value"]) if obs["value"] != "." else float("nan")
                for obs in observations
            },
            name=label,
            dtype=float,
        )
        s.index = pd.to_datetime(s.index)
        s.index.name = "date"

        logger.debug(f"  {series_id} ({label}): {len(s)} observations")
        return s

    def fetch(
        self,
        start_date: str = DEFAULT_START_DATE,
        end_date: str | None = None,
    ) -> None:
        """
        Fetch all series in self.series and save as a single wide parquet.

        We use a single persistent httpx.Client for connection reuse —
        this is more efficient than creating a new connection for each series.
        We also add a small sleep between requests to be polite to FRED's servers.
        """
        end = end_date or date.today().isoformat()
        logger.info(f"Fetching {len(self.series)} FRED series: {start_date} → {end}")

        series_list = []

        with httpx.Client() as client:
            for series_id, label in self.series.items():
                logger.info(f"  Fetching {series_id} → {label}")
                try:
                    s = self._fetch_one_series(series_id, label, start_date, end, client)
                    series_list.append(s)
                except Exception as e:
                    logger.warning(f"  Skipping {series_id}: {e}")
                # Be polite: wait 0.2 seconds between FRED API calls
                time.sleep(0.2)

        if not series_list:
            logger.error("No FRED series fetched successfully.")
            return

        # Combine all series into one wide DataFrame
        # pd.concat with axis=1 aligns on the date index automatically.
        # Different series have different frequencies (daily, monthly, quarterly)
        # so many cells will be NaN — that's expected and handled in processing.
        df = pd.concat(series_list, axis=1)
        df.index.name = "date"
        df = df.sort_index()
        df = df.reset_index()  # make date a column for parquet storage

        filename = f"macro_{datetime.now().strftime('%Y%m%d')}.parquet"
        output_path = self.raw_dir / filename
        df.to_parquet(output_path, index=False, engine="pyarrow")

        logger.info(
            f"Saved macro data: {len(df):,} rows × {len(self.series)} series "
            f"to {output_path}"
        )

    def load(self) -> pd.DataFrame:
        """Load the most recently fetched macro parquet file."""
        path = self._latest_file("macro_*.parquet")
        if path is None:
            raise FileNotFoundError(
                f"No macro data found in {self.raw_dir}. Run fetch() first."
            )
        logger.info(f"Loading macro data from {path}")
        return pd.read_parquet(path, engine="pyarrow")