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
  - We use an asof merge so each trading day gets the most recent
    available macro reading — this avoids look-ahead bias.

Output: data/processed/master/master_YYYYMMDD.parquet
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path
from datetime import datetime

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

        # ── Load processed macro data ─────────────────────────────────────────
        macro_path = self._latest(self.macro_dir, "macro_processed_*.parquet")
        logger.info(f"Loading macro data from {macro_path}")
        macro = pd.read_parquet(macro_path, engine="pyarrow")

        # ── Normalise datetime precision ──────────────────────────────────────
        # pyarrow can write timestamps as milliseconds (ms) or microseconds (us)
        # depending on the data source. pd.merge_asof requires both merge keys
        # to have identical dtypes — if one is ms and the other is us, it raises
        # an "incompatible merge keys" error even though the dates look the same.
        # Casting both to datetime64[us] guarantees they match before merging.
        equity["date"] = pd.to_datetime(equity["date"]).astype("datetime64[us]")
        macro["date"]  = pd.to_datetime(macro["date"]).astype("datetime64[us]")

        # ── Sort (required by merge_asof) ─────────────────────────────────────
        equity = equity.sort_values("date").reset_index(drop=True)
        macro  = macro.sort_values("date").reset_index(drop=True)

        # ── Asof merge ────────────────────────────────────────────────────────
        # merge_asof matches each equity date to the most recent macro date
        # that is <= the equity date.
        # Example: equity row dated 15 Jan gets the macro reading from 31 Dec.
        # The Feb macro reading is not used until Feb trading days arrive.
        # direction="backward" enforces this — no look-ahead bias.
        macro_cols = [c for c in macro.columns if c != "date"]

        master = pd.merge_asof(
            equity,
            macro[["date"] + macro_cols],
            on="date",
            direction="backward",
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