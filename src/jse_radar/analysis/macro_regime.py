"""
Macro Regime Detection — classify the macroeconomic environment by month.

What is a macro regime?
  A regime is a label that summarises the current state of the macro
  environment. Instead of using raw numbers (repo rate = 7.5%), we
  classify the *direction* and *level* of macro variables. This is
  more useful for analysis because equity behaviour often depends on
  whether conditions are getting better or worse, not the absolute level.

  Example: a repo rate of 7% falling from 8.5% is a very different
  environment to a repo rate of 7% rising from 3.5% — even though
  the number is the same.

We build four regime dimensions:

1. RATE REGIME
   Is the T-bill rate rising or falling?
   Rising = restrictive monetary policy (bad for growth stocks, REITs)
   Falling = accommodative monetary policy (good for growth, REITs)

2. INFLATION REGIME
   Is CPI YoY above or below the SARB target band (3-6%)?
   High inflation = SARB under pressure to hike, rand under pressure
   Low/target inflation = more policy flexibility

3. CURRENCY REGIME
   Is the rand strengthening or weakening over the past 3 months?
   Weak rand = headwind for importers, tailwind for rand-hedge stocks

4. COMPOSITE REGIME
   A single label combining rate + inflation for quick filtering:
   "HIKING_HIGH_INFLATION", "HIKING_LOW_INFLATION",
   "CUTTING_HIGH_INFLATION", "CUTTING_LOW_INFLATION"

These regime labels get joined back onto the master signals frame
so you can ask: "what was the average momentum score for gold miners
during CUTTING_LOW_INFLATION regimes?"
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from jse_radar.config import PROC_MACRO_DIR, PROC_MASTER_DIR
from jse_radar.utils.logger import get_logger

logger = get_logger(__name__)

# SARB inflation target band
SARB_TARGET_LOW  = 3.0
SARB_TARGET_HIGH = 6.0


class MacroRegimeClassifier:
    """
    Classifies each month into a macro regime based on processed FRED data.
    """

    def __init__(
        self,
        macro_dir:  Path = PROC_MACRO_DIR,
        master_dir: Path = PROC_MASTER_DIR,
    ) -> None:
        self.macro_dir  = macro_dir
        self.master_dir = master_dir

    def _latest(self, directory: Path, pattern: str) -> Path:
        files = sorted(directory.glob(pattern), key=lambda f: f.stat().st_mtime)
        if not files:
            raise FileNotFoundError(f"No files matching {pattern} in {directory}")
        return files[-1]

    def classify(self) -> pd.DataFrame:
        """
        Build a monthly regime DataFrame and return it.
        Also saves a regime-enriched master parquet.
        """
        logger.info("Classifying macro regimes...")

        # ── Load processed macro ──────────────────────────────────────────────
        macro_path = self._latest(self.macro_dir, "macro_processed_*.parquet")
        macro = pd.read_parquet(macro_path, engine="pyarrow")
        macro["date"] = pd.to_datetime(macro["date"])
        macro = macro.sort_values("date").reset_index(drop=True)

        logger.info(f"Loaded macro: {macro.shape}")

        # ── 1. Rate regime ────────────────────────────────────────────────────
        # Compare the current T-bill rate to its 3-month rolling mean.
        # If current rate > 3M mean: rates are rising → HIKING
        # If current rate < 3M mean: rates are falling → CUTTING
        # We use a 3-month window to smooth out noise from missing months.

        if "tbill_rate" in macro.columns:
            macro["rate_3m_avg"] = macro["tbill_rate"].rolling(3, min_periods=2).mean()
            macro["rate_regime"] = np.where(
                macro["tbill_rate"] > macro["rate_3m_avg"], "HIKING",
                np.where(
                    macro["tbill_rate"] < macro["rate_3m_avg"], "CUTTING",
                    "NEUTRAL"
                )
            )
        else:
            logger.warning("tbill_rate not found — rate regime will be UNKNOWN")
            macro["rate_regime"] = "UNKNOWN"

        # ── 2. Inflation regime ───────────────────────────────────────────────
        # Compare CPI YoY to the SARB target band (3-6%).
        # Above 6%: HIGH_INFLATION (SARB under pressure)
        # 3-6%:     TARGET_INFLATION (within band)
        # Below 3%: LOW_INFLATION (deflation risk, room to cut)

        if "cpi_yoy_pct" in macro.columns:
            macro["inflation_regime"] = pd.cut(
                macro["cpi_yoy_pct"],
                bins=[-np.inf, SARB_TARGET_LOW, SARB_TARGET_HIGH, np.inf],
                labels=["LOW_INFLATION", "TARGET_INFLATION", "HIGH_INFLATION"],
            ).astype(str)
        else:
            logger.warning("cpi_yoy_pct not found — inflation regime will be UNKNOWN")
            macro["inflation_regime"] = "UNKNOWN"

        # ── 3. Currency regime ────────────────────────────────────────────────
        # 3-month cumulative ZAR/USD change.
        # Positive (rand weakened) → WEAK_RAND
        # Negative (rand strengthened) → STRONG_RAND

        if "zar_usd" in macro.columns:
            macro["zar_3m_change"] = macro["zar_usd"].pct_change(3) * 100
            macro["currency_regime"] = np.where(
                macro["zar_3m_change"] >  2,  "WEAK_RAND",
                np.where(
                    macro["zar_3m_change"] < -2, "STRONG_RAND",
                    "STABLE_RAND"
                )
            )
        else:
            logger.warning("zar_usd not found — currency regime will be UNKNOWN")
            macro["currency_regime"] = "UNKNOWN"

        # ── 4. Composite regime ───────────────────────────────────────────────
        # Combine rate + inflation into one label.
        # This is the primary regime label used in analysis.
        macro["composite_regime"] = (
            macro["rate_regime"] + "_" + macro["inflation_regime"]
        )

        # ── 5. Regime duration ────────────────────────────────────────────────
        # How many consecutive months has the current composite regime been active?
        # This tells you whether you're early or late in a regime.
        macro["regime_changed"] = (
            macro["composite_regime"] != macro["composite_regime"].shift(1)
        ).astype(int)
        macro["regime_duration"] = (
            macro.groupby(macro["regime_changed"].cumsum())
            .cumcount() + 1
        )

        logger.info("Regime distribution:")
        logger.info(f"\n{macro['composite_regime'].value_counts().to_string()}")

        # ── Save regime table ─────────────────────────────────────────────────
        regime_cols = [
            "date", "tbill_rate", "cpi_yoy_pct", "zar_usd",
            "rate_regime", "inflation_regime", "currency_regime",
            "composite_regime", "regime_duration",
        ]
        # Keep only columns that exist
        regime_cols = [c for c in regime_cols if c in macro.columns]
        regime_df = macro[regime_cols].copy()

        filename = f"macro_regimes_{datetime.now().strftime('%Y%m%d')}.parquet"
        out_path  = self.macro_dir / filename
        regime_df.to_parquet(out_path, index=False, engine="pyarrow")
        logger.info(f"Regime table saved → {out_path}")

        # ── Join regimes onto master signals frame ────────────────────────────
        # Find the latest signals file (written by SignalEngine)
        # Fall back to master file if signals haven't been run yet
        signal_files = sorted(
            self.master_dir.glob("master_signals_*.parquet"),
            key=lambda f: f.stat().st_mtime,
        )
        master_files = sorted(
            self.master_dir.glob("master_*.parquet"),
            key=lambda f: f.stat().st_mtime,
        )

        if signal_files:
            master_path = signal_files[-1]
        elif master_files:
            master_path = master_files[-1]
        else:
            raise FileNotFoundError("No master parquet found. Run pipeline first.")

        logger.info(f"Loading master frame from {master_path}")
        master = pd.read_parquet(master_path, engine="pyarrow")
        master["date"] = pd.to_datetime(master["date"]).astype("datetime64[us]")
        regime_df["date"] = regime_df["date"].astype("datetime64[us]")

        # Asof merge — same logic as master builder:
        # each equity date gets the most recent regime classification
        master = master.sort_values("date")
        regime_df = regime_df.sort_values("date")

        regime_merge_cols = [
            "date", "rate_regime", "inflation_regime",
            "currency_regime", "composite_regime", "regime_duration",
        ]
        regime_merge_cols = [c for c in regime_merge_cols if c in regime_df.columns]

        master_enriched = pd.merge_asof(
            master,
            regime_df[regime_merge_cols],
            on="date",
            direction="backward",
        )

        enriched_filename = f"master_regimes_{datetime.now().strftime('%Y%m%d')}.parquet"
        enriched_path     = self.master_dir / enriched_filename
        master_enriched.to_parquet(enriched_path, index=False, engine="pyarrow")

        logger.info(
            f"Regime-enriched master saved → {enriched_path} "
            f"({master_enriched.shape})"
        )
        return regime_df