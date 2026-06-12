"""
Equity Processor — cleans raw equity data and engineers features.

What we do here:
1. Load the raw parquet saved by EquityFetcher
2. Handle missing values (forward-fill within each ticker)
3. Calculate daily returns, log returns, rolling volatility
4. Add calendar features (day of week, month, quarter)
5. Separate indices from individual stocks
6. Save clean data to data/processed/equities/

Why these features?
  - Daily returns: the bread and butter of equity analysis
  - Log returns: better for statistical modelling (additive, more normal)
  - Rolling volatility (21-day): proxy for market risk
  - These are the inputs to any signal-generation model we'll build later
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from jse_radar.config import RAW_EQUITY_DIR, PROC_EQUITY_DIR
from jse_radar.data.equity_fetcher import EquityFetcher
from jse_radar.utils.logger import get_logger

logger = get_logger(__name__)


class EquityProcessor:
    """Cleans and enriches raw JSE equity data."""

    def __init__(
        self,
        raw_dir: Path = RAW_EQUITY_DIR,
        processed_dir: Path = PROC_EQUITY_DIR,
    ) -> None:
        self.raw_dir       = raw_dir
        self.processed_dir = processed_dir
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def process(self) -> pd.DataFrame:
        """
        Full processing pipeline. Returns the processed DataFrame
        and saves it to parquet.
        """
        logger.info("Starting equity processing...")

        # ── 1. Load raw data ──────────────────────────────────────────────────
        fetcher = EquityFetcher(raw_dir=self.raw_dir)
        df = fetcher.load()
        logger.info(f"Loaded {len(df):,} raw rows")

        # ── 2. Ensure correct dtypes ──────────────────────────────────────────
        df["date"] = pd.to_datetime(df["date"])
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # ── 3. Sort and set index ─────────────────────────────────────────────
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

        # ── 4. Forward-fill missing prices within each ticker ────────────────
        # On days where a stock doesn't trade (e.g. suspended), yfinance may
        # return NaN. We forward-fill so we always have a price — but cap at
        # 5 consecutive days to avoid stale data propagating too far.
        df["close"] = (
            df.groupby("ticker")["close"]
            .transform(lambda s: s.ffill(limit=5))
        )

        # ── 5. Calculate returns ──────────────────────────────────────────────
        # shift(1) gives the previous day's close within each ticker group.
        # pct_change() on the grouped close gives the daily return.
        df["daily_return"] = (
            df.groupby("ticker")["close"]
            .pct_change()
        )

        # Log return: ln(close_t / close_{t-1})
        # Better for modelling: approximately equal to daily_return for small moves,
        # but additive over time (monthly log return = sum of daily log returns)
        df["log_return"] = (
            df.groupby("ticker")["close"]
            .transform(lambda s: np.log(s / s.shift(1)))
        )

        # ── 6. Rolling volatility (annualised) ────────────────────────────────
        # Standard deviation of daily log returns over 21 trading days (~1 month)
        # Multiply by sqrt(252) to annualise (252 trading days per year)
        df["volatility_21d"] = (
            df.groupby("ticker")["log_return"]
            .transform(lambda s: s.rolling(window=21, min_periods=10).std() * np.sqrt(252))
        )

        # ── 7. Rolling 52-week high/low (for momentum signals) ───────────────
        df["high_52w"] = (
            df.groupby("ticker")["close"]
            .transform(lambda s: s.rolling(window=252, min_periods=50).max())
        )
        df["low_52w"] = (
            df.groupby("ticker")["close"]
            .transform(lambda s: s.rolling(window=252, min_periods=50).min())
        )

        # Distance from 52-week high (0 = at high, -1 = at low)
        df["pct_from_52w_high"] = (df["close"] - df["high_52w"]) / df["high_52w"]

        # ── 8. Calendar features ──────────────────────────────────────────────
        df["year"]        = df["date"].dt.year
        df["month"]       = df["date"].dt.month
        df["quarter"]     = df["date"].dt.quarter
        df["day_of_week"] = df["date"].dt.dayofweek   # 0=Monday, 4=Friday

        # ── 9. Save ───────────────────────────────────────────────────────────
        filename = f"equities_processed_{datetime.now().strftime('%Y%m%d')}.parquet"
        output_path = self.processed_dir / filename
        df.to_parquet(output_path, index=False, engine="pyarrow")

        logger.info(
            f"Equity processing complete. "
            f"{len(df):,} rows, {df['ticker'].nunique()} tickers → {output_path}"
        )
        return df