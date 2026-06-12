"""
Master Builder — joins equity and macro data into one analytical frame.

This is the final step in the pipeline. It produces a single wide
DataFrame where every row is a (date, ticker) combination and every
column is either an equity metric or a macro indicator valid for that date.

Why join at the ticker level?
  Macro conditions (repo rate, CPI, ZAR/USD) apply to every stock equally,
  but we want one row per stock per day so that in the analysis layer we can
  ask: "how does this stock's return correlate with the repo rate?"

Join logic:
  - Equity data is daily
  - Macro data is monthly (after processing)
  - We merge on year + month (asof merge so each trading day gets the
    most recent available macro reading — this avoids look-ahead bias)

Output: data/processed/master/master_YYYYMMDD.parquet
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path
from datetime import datetime
import glob

from jse_radar.config import PROC_EQUITY_DIR, PROC_MACRO_DIR, PROC_MASTER_DIR
from jse_radar.utils.logger import get_logger

logger = get_logger(__name__)


class MasterBuilder:
    """Joins processed equity and macro data into a single analytical frame."""

    def __init__(
        self,
        equity_dir: Path = PROC_EQUITY_DIR,
        macro_dir: Path = PROC_MACRO_DIR,
        master_dir: Path = PROC_MASTER_DIR,
    ) -> None:
        self.equity_dir = equity_dir
        self.macro_dir  = macro_dir
        self.master_dir = master_dir
        self.master_dir.mkdir(parents=True, exist_ok=True)

    def _latest(self, directory: Path, pattern: str) -> Path:
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
        equity["date"] = pd.to_datetime(equity["date"])

        # ── Load processed macro data ─────────────────────────────────────────
        macro_path = self._latest(self.macro_dir, "macro_processed_*.parquet")
        logger.info(f"Loading macro data from {macro_path}")
        macro = pd.read_parquet(macro_path, engine="pyarrow")
        macro["date"] = pd.to_datetime(macro["date"])

        # ── Prepare for asof merge ────────────────────────────────────────────
        # pd.merge_asof matches each equity date to the most recent macro date
        # that is <= the equity date. This is correct: on 15 Jan, you use the
        # macro reading from 31 Dec (last month-end). You don't use Feb data
        # because you wouldn't know that yet — no look-ahead bias.
        equity = equity.sort_values("date")
        macro  = macro.sort_values("date")

        # Macro columns to bring across (exclude 'date' which is the merge key)
        macro_cols = [c for c in macro.columns if c != "date"]

        master = pd.merge_asof(
            equity,
            macro[["date"] + macro_cols],
            on="date",
            direction="backward",   # use the most recent macro reading on or before equity date
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