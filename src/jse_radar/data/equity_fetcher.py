"""
JSE Equity Fetcher — downloads JSE stock prices via yfinance.

How yfinance works with JSE:
  Yahoo Finance uses the ".JO" suffix for JSE stocks.
  "SOL.JO" is Sasol, "NPN.JO" is Naspers, "^J203.JO" is the ALSI.
  yfinance fetches daily OHLCV (Open, High, Low, Close, Volume)
  plus Adjusted Close (which accounts for dividends and splits).

What we save:
  One parquet file per run, named by date:
  data/raw/equities/equities_YYYYMMDD.parquet

  Parquet is a columnar binary format — much smaller than CSV and
  much faster to read. A 20-year daily dataset for 30 tickers
  that would be ~50MB as CSV is ~5MB as parquet.

Column structure of saved file:
  date | ticker | open | high | low | close | adj_close | volume | name
"""

from __future__ import annotations

from datetime import datetime, date
from pathlib import Path
import pandas as pd
import yfinance as yf

from jse_radar.config import JSE_TICKERS, RAW_EQUITY_DIR, DEFAULT_START_DATE
from jse_radar.data.fetcher import DataFetcher
from jse_radar.utils.logger import get_logger

logger = get_logger(__name__)


class EquityFetcher(DataFetcher):
    """Fetches JSE equity and index price data from Yahoo Finance."""

    def __init__(
        self,
        tickers: dict[str, str] | None = None,
        raw_dir: Path = RAW_EQUITY_DIR,
    ) -> None:
        super().__init__(raw_dir)
        # Use the default universe from config if none provided
        # This allows overriding for testing: EquityFetcher(tickers={"SOL.JO": "Sasol"})
        self.tickers = tickers or JSE_TICKERS

    def fetch(
        self,
        start_date: str = DEFAULT_START_DATE,
        end_date: str | None = None,
    ) -> None:
        """
        Download OHLCV data for all tickers in self.tickers.

        yfinance's download() function is vectorised — it fetches all
        tickers in one HTTP request rather than one request per ticker.
        This is much faster and more polite to Yahoo's servers.

        The returned DataFrame has a MultiIndex on columns:
          (Price, Ticker) e.g. ("Close", "SOL.JO")
        We reshape (melt) this into a long format with one row per
        date-ticker combination, which is much easier to work with.
        """
        end = end_date or date.today().isoformat()
        logger.info(f"Fetching {len(self.tickers)} JSE tickers: {start_date} → {end}")

        ticker_list = list(self.tickers.keys())

        try:
            # auto_adjust=True means Close is already dividend/split adjusted.
            # actions=False means we don't download the dividend/split table
            # (we can add that later as a separate fetch).
            raw = yf.download(
                tickers=ticker_list,
                start=start_date,
                end=end,
                auto_adjust=True,
                actions=False,
                progress=False,   # suppress the tqdm progress bar
                threads=True,     # parallel downloads
            )
        except Exception as e:
            logger.error(f"yfinance download failed: {e}")
            raise

        if raw.empty:
            logger.warning("yfinance returned an empty DataFrame. Check tickers.")
            return

        # ── Reshape from wide MultiIndex to long format ───────────────────────
        # raw.columns looks like: MultiIndex([('Close', 'SOL.JO'), ('Open', 'SOL.JO'), ...])
        # stack() pivots the ticker level down into a row index, giving us:
        #   date | ticker | Close | High | Low | Open | Volume
        df = raw.stack(level=1, future_stack=True).reset_index()
        df.columns.name = None

        # Rename columns to our standard schema
        df = df.rename(columns={
            "Date":   "date",
            "Ticker": "ticker",
            "Open":   "open",
            "High":   "high",
            "Low":    "low",
            "Close":  "close",
            "Volume": "volume",
        })

        # Add the human-readable name from our ticker dict
        df["name"] = df["ticker"].map(self.tickers)

        # Ensure date column is proper datetime (not datetime with timezone)
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

        # Drop rows where close price is NaN (missing trading days, delisted, etc.)
        before = len(df)
        df = df.dropna(subset=["close"])
        dropped = before - len(df)
        if dropped:
            logger.debug(f"Dropped {dropped} rows with NaN close prices")

        # Sort so the data is chronological per ticker
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

        # ── Save to parquet ───────────────────────────────────────────────────
        filename = f"equities_{datetime.now().strftime('%Y%m%d')}.parquet"
        output_path = self.raw_dir / filename
        df.to_parquet(output_path, index=False, engine="pyarrow")

        logger.info(
            f"Saved {len(df):,} rows × {df['ticker'].nunique()} tickers "
            f"to {output_path}"
        )

    def load(self) -> pd.DataFrame:
        """
        Load the most recently fetched equity parquet file.

        We use _latest_file() from the base class to find it —
        this means you don't need to know the exact filename.
        """
        path = self._latest_file("equities_*.parquet")
        if path is None:
            raise FileNotFoundError(
                f"No equity data found in {self.raw_dir}. Run fetch() first."
            )
        logger.info(f"Loading equity data from {path}")
        return pd.read_parquet(path, engine="pyarrow")