"""
Macro Processor — cleans and standardises raw FRED macro data.

What we do:
1. Load raw FRED parquet
2. Set a proper DatetimeIndex
3. Resample mixed-frequency data to monthly (the LCD of our series)
4. Forward-fill sparse series (repo rate changes monthly, not daily)
5. Calculate year-on-year and month-on-month changes for flow variables
6. Save to data/processed/macro/
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path
from datetime import datetime

from jse_radar.config import RAW_MACRO_DIR, PROC_MACRO_DIR
from jse_radar.data.macro_fetcher import MacroFetcher
from jse_radar.utils.logger import get_logger

logger = get_logger(__name__)


class MacroProcessor:
    """Cleans and standardises raw FRED macroeconomic data."""

    def __init__(
        self,
        raw_dir: Path = RAW_MACRO_DIR,
        processed_dir: Path = PROC_MACRO_DIR,
    ) -> None:
        self.raw_dir       = raw_dir
        self.processed_dir = processed_dir
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def process(self) -> pd.DataFrame:
        """Full processing pipeline."""
        logger.info("Starting macro processing...")

        # ── 1. Load ───────────────────────────────────────────────────────────
        fetcher = MacroFetcher.__new__(MacroFetcher)
        fetcher.raw_dir = self.raw_dir
        df = MacroFetcher(raw_dir=self.raw_dir, api_key="dummy").load() \
            if False else self._load_raw()

        # ── 2. Set DatetimeIndex ──────────────────────────────────────────────
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()

        # ── 3. Resample to month-end frequency ───────────────────────────────
        # FRED series have different frequencies:
        #   - zar_usd: daily
        #   - cpi_all: monthly
        #   - gdp_constant: quarterly
        # We resample everything to month-end using last() for rates/levels
        # and mean() for the exchange rate (monthly average is more meaningful
        # than a single end-of-month snapshot for FX).
        fx_cols = ["zar_usd", "usd_eur"]
        other_cols = [c for c in df.columns if c not in fx_cols]

        monthly_fx    = df[fx_cols].resample("ME").mean()
        monthly_other = df[other_cols].resample("ME").last()
        df_monthly = pd.concat([monthly_fx, monthly_other], axis=1)

        # ── 4. Forward-fill sparse series ────────────────────────────────────
        # The repo rate only changes when the MPC meets (~every 2 months).
        # Between meetings, FRED has NaN. Forward-fill with a 3-month cap.
        df_monthly = df_monthly.ffill(limit=3)

        # ── 5. Derived indicators ─────────────────────────────────────────────
        # Month-on-month change in ZAR/USD (positive = rand weakened)
        if "zar_usd" in df_monthly.columns:
            df_monthly["zar_usd_mom_pct"] = df_monthly["zar_usd"].pct_change() * 100

        # Year-on-year change in CPI (this IS the inflation rate, cross-check FRED)
        if "cpi_all" in df_monthly.columns:
            df_monthly["cpi_yoy_pct"] = df_monthly["cpi_all"].pct_change(12) * 100

        # Real interest rate (repo rate minus CPI YoY inflation)
        if "tbill_rate" in df_monthly.columns and "cpi_yoy_pct" in df_monthly.columns:
            df_monthly["real_tbill_rate"] = (
        df_monthly["tbill_rate"] - df_monthly["cpi_yoy_pct"]
    )

# Yield curve spread (10y bond minus T-bill) — positive = normal, negative = inverted
        if "govt_bond_10y" in df_monthly.columns and "tbill_rate" in df_monthly.columns:
             df_monthly["yield_curve_spread"] = (
        df_monthly["govt_bond_10y"] - df_monthly["tbill_rate"]
    )

        # ── 6. Reset index and save ───────────────────────────────────────────
        df_monthly = df_monthly.reset_index()

        filename = f"macro_processed_{datetime.now().strftime('%Y%m%d')}.parquet"
        output_path = self.processed_dir / filename
        df_monthly.to_parquet(output_path, index=False, engine="pyarrow")

        logger.info(
            f"Macro processing complete. {len(df_monthly):,} monthly rows "
            f"→ {output_path}"
        )
        return df_monthly

    def _load_raw(self) -> pd.DataFrame:
        """Load raw macro parquet without needing a live API key."""
        from jse_radar.data.fetcher import DataFetcher
        import glob

        files = sorted(
            (self.raw_dir).glob("macro_*.parquet"),
            key=lambda f: f.stat().st_mtime,
        )
        if not files:
            raise FileNotFoundError(
                f"No macro raw data in {self.raw_dir}. Run MacroFetcher.fetch() first."
            )
        logger.info(f"Loading raw macro from {files[-1]}")
        return pd.read_parquet(files[-1], engine="pyarrow")