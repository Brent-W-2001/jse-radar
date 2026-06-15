"""
Rolling Correlation Analysis — how equity returns relate to macro variables over time.

Why rolling correlations?
  A single static correlation (like the bar chart in the notebook) tells you
  the average relationship over the entire period. But relationships change.
  Gold miners might be highly correlated with ZAR/USD during a commodity
  supercycle but less so during a global risk-off event when everything
  sells off together.

  Rolling correlations show you HOW and WHEN the relationship changes.
  A 90-day rolling window means: "what was the correlation between this
  stock and ZAR/USD over the past 90 trading days?"

What we compute:
  For each ticker, we compute 90-day rolling correlations between
  daily_return and each macro variable in the master frame.
  The output is a long-format DataFrame: one row per (date, ticker, macro_var)
  with the correlation value.

Why 90 days?
  - Short enough to detect regime changes
  - Long enough to be statistically meaningful (need ~30+ observations)
  - Corresponds roughly to one quarter — a natural business cycle unit

Output format (long):
  date | ticker | macro_var | rolling_corr

This format is ideal for Plotly — you can filter by ticker or macro_var
and plot how the relationship evolves over time.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from jse_radar.config import PROC_MASTER_DIR
from jse_radar.utils.logger import get_logger

logger = get_logger(__name__)

# Macro columns to correlate against equity returns
# These must exist in the master frame
MACRO_VARS = [
    "zar_usd_mom_pct",   # Monthly ZAR/USD change
    "tbill_rate",         # T-bill rate level
    "cpi_yoy_pct",        # CPI year-on-year
    "real_tbill_rate",    # Real interest rate
    "exports_value",      # Export activity
    "imports_value",      # Import activity
]

ROLLING_WINDOW = 90   # trading days


class CorrelationAnalyser:
    """Computes rolling correlations between equity returns and macro variables."""

    def __init__(self, master_dir: Path = PROC_MASTER_DIR) -> None:
        self.master_dir = master_dir

    def _latest(self, pattern: str) -> Path:
        files = sorted(
            self.master_dir.glob(pattern),
            key=lambda f: f.stat().st_mtime,
        )
        if not files:
            raise FileNotFoundError(
                f"No files matching {pattern} in {self.master_dir}"
            )
        return files[-1]

    def compute(self) -> pd.DataFrame:
        """
        Compute rolling correlations and return a long-format DataFrame.
        """
        logger.info("Computing rolling macro correlations...")

        # ── Load the most enriched master available ───────────────────────────
        # Prefer regime-enriched > signals > plain master
        for pattern in [
            "master_regimes_*.parquet",
            "master_signals_*.parquet",
            "master_*.parquet",
        ]:
            try:
                path = self._latest(pattern)
                break
            except FileNotFoundError:
                continue
        else:
            raise FileNotFoundError("No master frame found. Run pipeline first.")

        logger.info(f"Loading from {path}")
        df = pd.read_parquet(path, engine="pyarrow")
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

        # ── Identify which macro vars actually exist in this frame ────────────
        available_vars = [v for v in MACRO_VARS if v in df.columns]
        missing_vars   = [v for v in MACRO_VARS if v not in df.columns]
        if missing_vars:
            logger.warning(f"Macro vars not found in master frame: {missing_vars}")
        logger.info(f"Computing correlations for: {available_vars}")

        # ── Compute rolling correlations per ticker ───────────────────────────
        # Strategy: for each ticker, extract its time series, then for each
        # macro variable compute the rolling correlation with daily_return.
        # Collect results into a list of DataFrames, then concatenate.

        results = []
        tickers = df["ticker"].unique()

        for ticker in tickers:
            tk_df = df[df["ticker"] == ticker].copy()

            for macro_var in available_vars:
                # rolling().corr() computes Pearson correlation over the window.
                # min_periods=30 means we need at least 30 non-NaN pairs —
                # below that we return NaN rather than a noisy estimate.
                rolling_corr = (
                    tk_df["daily_return"]
                    .rolling(window=ROLLING_WINDOW, min_periods=30)
                    .corr(tk_df[macro_var])
                )

                result = pd.DataFrame({
                    "date":        tk_df["date"].values,
                    "ticker":      ticker,
                    "name":        tk_df["name"].iloc[0] if "name" in tk_df.columns else ticker,
                    "macro_var":   macro_var,
                    "rolling_corr": rolling_corr.values,
                })
                results.append(result)

        corr_df = pd.concat(results, ignore_index=True)
        corr_df = corr_df.dropna(subset=["rolling_corr"])
        corr_df = corr_df.sort_values(["ticker", "macro_var", "date"])

        logger.info(f"Rolling correlation frame: {corr_df.shape}")

        # ── Save ──────────────────────────────────────────────────────────────
        filename = f"rolling_correlations_{datetime.now().strftime('%Y%m%d')}.parquet"
        out_path = self.master_dir / filename
        corr_df.to_parquet(out_path, index=False, engine="pyarrow")
        logger.info(f"Saved → {out_path}")

        # ── Summary: average correlation per ticker-macro pair ────────────────
        summary = (
            corr_df.groupby(["ticker", "macro_var"])["rolling_corr"]
            .agg(["mean", "std", "min", "max"])
            .round(4)
        )
        logger.info(f"\nCorrelation summary:\n{summary.to_string()}")

        return corr_df

    def static_correlation_matrix(self) -> pd.DataFrame:
        """
        Return a static (full-period) correlation matrix between
        all tickers' daily returns. Useful for portfolio construction
        — highly correlated stocks don't add diversification.
        """
        for pattern in [
            "master_regimes_*.parquet",
            "master_signals_*.parquet",
            "master_*.parquet",
        ]:
            try:
                path = self._latest(pattern)
                break
            except FileNotFoundError:
                continue

        df = pd.read_parquet(path, engine="pyarrow")

        # Pivot to wide format: date × ticker, values = daily_return
        wide = df.pivot_table(
            index="date",
            columns="ticker",
            values="daily_return",
        )

        corr_matrix = wide.corr()
        logger.info(f"Static correlation matrix: {corr_matrix.shape}")
        return corr_matrix