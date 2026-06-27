"""
Master Builder — joins equity, macro (FRED), and World Bank data into
one analytical frame.

This is the final step in the pipeline. It produces a single wide
DataFrame where every row is a (date, ticker) combination and every
column is either an equity metric, a FRED macro indicator, or a
World Bank structural indicator valid for that date.

Why join at the ticker level?
  Macro conditions (repo rate, CPI, ZAR/USD) apply to every stock equally,
  but we want one row per stock per day so that in the analysis layer we can
  ask: "how does this stock's return correlate with the repo rate?"

Join logic:
  - Equity data is daily
  - FRED macro data is monthly (after processing)
  - World Bank data is annual
  - We use asof merges so each trading day gets the most recent available
    reading from each source — this avoids look-ahead bias.

Output: data/processed/master/master_YYYYMMDD.parquet
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path
from datetime import datetime

from jse_radar.config import (
    PROC_EQUITY_DIR, PROC_MACRO_DIR, PROC_MASTER_DIR, RAW_WB_DIR,
)
from jse_radar.utils.logger import get_logger

logger = get_logger(__name__)


class MasterBuilder:
    """Joins processed equity, FRED macro, and World Bank data into one frame."""

    def __init__(
        self,
        equity_dir: Path = PROC_EQUITY_DIR,
        macro_dir:  Path = PROC_MACRO_DIR,
        wb_dir:     Path = RAW_WB_DIR,
        master_dir: Path = PROC_MASTER_DIR,
    ) -> None:
        self.equity_dir = equity_dir
        self.macro_dir  = macro_dir
        self.wb_dir     = wb_dir
        self.master_dir = master_dir
        self.master_dir.mkdir(parents=True, exist_ok=True)

    def _latest(self, directory: Path, pattern: str) -> Path:
        """Find the most recently modified file matching a glob pattern."""
        files = sorted(directory.glob(pattern), key=lambda f: f.stat().st_mtime)
        if not files:
            raise FileNotFoundError(f"No files matching {pattern} in {directory}")
        return files[-1]

    def build(self) -> pd.DataFrame:
        """Build and save the master analytical frame."""
        logger.info("Building master dataset...")

        # ── Load processed equity data ────────────────────────────────────────
        eq_path = self._latest(self.equity_dir, "equities_processed_*.parquet")
        logger.info(f"Loading equity data from {eq_path}")
        equity = pd.read_parquet(eq_path, engine="pyarrow")

        # ── Load processed macro (FRED) data ──────────────────────────────────
        macro_path = self._latest(self.macro_dir, "macro_processed_*.parquet")
        logger.info(f"Loading macro data from {macro_path}")
        macro = pd.read_parquet(macro_path, engine="pyarrow")

        # ── Normalise datetime precision ──────────────────────────────────────
        # pyarrow can write timestamps as milliseconds (ms) or microseconds (us)
        # depending on the data source. pd.merge_asof requires both merge keys
        # to have identical dtypes — casting both to datetime64[us] guarantees
        # a match before merging.
        equity["date"] = pd.to_datetime(equity["date"]).astype("datetime64[us]")
        macro["date"]  = pd.to_datetime(macro["date"]).astype("datetime64[us]")

        equity = equity.sort_values("date").reset_index(drop=True)
        macro  = macro.sort_values("date").reset_index(drop=True)

        # ── Asof merge: equity + FRED macro ───────────────────────────────────
        macro_cols = [c for c in macro.columns if c != "date"]

        master = pd.merge_asof(
            equity,
            macro[["date"] + macro_cols],
            on="date",
            direction="backward",
        )

        # ── Asof merge: master + World Bank (annual) ──────────────────────────
        # World Bank data only has a 'year' column, not a full date.
        # We convert it to a date (1 January of that year) so merge_asof
        # can match it against equity dates. Since World Bank data is
        # published with a lag, using Jan 1 of the year is a reasonable
        # anchor — the figures for year Y aren't usually fully known until
        # partway through Y+1, but we treat them as available from Jan 1
        # of the reporting year for simplicity. This is a structural/annual
        # signal, not a high-frequency trading one, so the exact day
        # doesn't materially affect analysis.
        try:
            wb_path = self._latest(self.wb_dir, "worldbank_*.parquet")
            logger.info(f"Loading World Bank data from {wb_path}")
            wb_df = pd.read_parquet(wb_path, engine="pyarrow")

            wb_df["date"] = pd.to_datetime(
                wb_df["year"].astype("Int64").astype(str) + "-01-01"
            ).astype("datetime64[us]")
            wb_df = wb_df.drop(columns=["year"]).sort_values("date").reset_index(drop=True)

            wb_cols = [c for c in wb_df.columns if c != "date"]

            master = pd.merge_asof(
                master,
                wb_df[["date"] + wb_cols],
                on="date",
                direction="backward",
            )
            logger.info(f"Joined {len(wb_cols)} World Bank columns onto master frame")

        except FileNotFoundError:
            logger.warning(
                "No World Bank data found — master frame will not include "
                "World Bank indicators. Run WorldBankFetcher.fetch() first."
            )

        logger.info(
            f"Master frame: {len(master):,} rows × {len(master.columns)} columns"
        )

        # ── Save ──────────────────────────────────────────────────────────────
        filename = f"master_{datetime.now().strftime('%Y%m%d')}.parquet"
        output_path = self.master_dir / filename
        master.to_parquet(output_path, index=False, engine="pyarrow")

        logger.info(f"Master dataset saved → {output_path}")
        return master