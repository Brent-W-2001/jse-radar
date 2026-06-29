"""
Entry point for the full jse-radar data pipeline.

Run from the repo root with the jse-radar conda environment active:

    conda activate jse-radar
    python scripts/run_pipeline.py

Or with custom dates:

    python scripts/run_pipeline.py --start 2015-01-01

The script runs all three fetchers, then all processors, then the master
builder, then the analysis layer (signals, regimes, correlations), then
a final data quality check. It prints a summary at the end so you know
what succeeded.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add the repo root to sys.path so we can import jse_radar
# (this is only needed when running as a script, not when installed via pip -e .)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jse_radar.utils.logger import get_logger
from jse_radar.utils.data_quality import run_all_checks

logger = get_logger("run_pipeline")


def parse_args():
    parser = argparse.ArgumentParser(description="Run the jse-radar data pipeline")
    parser.add_argument(
        "--start", default="2000-01-01",
        help="Start date (YYYY-MM-DD). Default: 2000-01-01"
    )
    parser.add_argument(
        "--end", default=None,
        help="End date (YYYY-MM-DD). Default: today"
    )
    parser.add_argument(
        "--skip-fetch", action="store_true",
        help="Skip fetching (use existing raw data)"
    )
    parser.add_argument(
        "--skip-process", action="store_true",
        help="Skip processing (use existing processed data)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("=" * 60)
    logger.info("jse-radar pipeline starting")
    logger.info(f"  Date range: {args.start} → {args.end or 'today'}")
    logger.info("=" * 60)

    # ── Phase 1: Fetch ────────────────────────────────────────────────────────
    if not args.skip_fetch:
        from jse_radar.data.pipeline import run as fetch_all
        fetch_results = fetch_all(start_date=args.start, end_date=args.end)
        if not any(fetch_results.values()):
            logger.error("All fetchers failed. Aborting.")
            sys.exit(1)
    else:
        logger.info("Skipping fetch phase (--skip-fetch)")

    # ── Phase 2: Process ──────────────────────────────────────────────────────
    if not args.skip_process:
        from jse_radar.processing.equity_processor import EquityProcessor
        from jse_radar.processing.macro_processor import MacroProcessor

        try:
            EquityProcessor().process()
        except Exception as e:
            logger.error(f"Equity processing failed: {e}")

        try:
            MacroProcessor().process()
        except Exception as e:
            logger.error(f"Macro processing failed: {e}")
    else:
        logger.info("Skipping process phase (--skip-process)")

    # ── Phase 3: Build master ─────────────────────────────────────────────────
    try:
        from jse_radar.processing.master_builder import MasterBuilder
        master = MasterBuilder().build()
        logger.info(f"Master frame shape: {master.shape}")
    except Exception as e:
        logger.error(f"Master build failed: {e}")

    # ── Phase 4: Analysis ─────────────────────────────────────────────────────
    try:
        from jse_radar.analysis.signals import SignalEngine
        SignalEngine().compute()
    except Exception as e:
        logger.error(f"Signal computation failed: {e}")

    try:
        from jse_radar.analysis.macro_regime import MacroRegimeClassifier
        MacroRegimeClassifier().classify()
    except Exception as e:
        logger.error(f"Macro regime classification failed: {e}")

    try:
        from jse_radar.analysis.correlation import CorrelationAnalyser
        CorrelationAnalyser().compute()
    except Exception as e:
        logger.error(f"Correlation analysis failed: {e}")

    # ── Phase 5: Data quality checks ──────────────────────────────────────────
    # Runs last, after every other stage has completed. Never blocks or
    # raises — only logs warnings/errors for a human to review. This is
    # the safety net that catches the next silent data issue automatically,
    # instead of relying on someone noticing a gap in a dashboard chart.
    try:
        from jse_radar.config import RAW_EQUITY_DIR, RAW_MACRO_DIR, PROC_MASTER_DIR

        equity_df_for_check   = None
        macro_df_for_check    = None
        master_df_for_check   = None
        previous_master_path  = None

        # Load whatever's currently on disk for each stage — these are
        # the same files just written earlier in this same pipeline run
        equity_files = sorted(RAW_EQUITY_DIR.glob("equities_*.parquet"))
        if equity_files:
            equity_df_for_check = pd.read_parquet(equity_files[-1], engine="pyarrow")

        macro_files = sorted(RAW_MACRO_DIR.glob("macro_*.parquet"))
        if macro_files:
            macro_df_for_check = pd.read_parquet(macro_files[-1], engine="pyarrow")

        master_files = sorted(PROC_MASTER_DIR.glob("master_2*.parquet"))
        if len(master_files) >= 1:
            master_df_for_check = pd.read_parquet(master_files[-1], engine="pyarrow")
        if len(master_files) >= 2:
            # The second-to-last file is the previous run's master frame
            previous_master_path = master_files[-2]

        run_all_checks(
            equity_df=equity_df_for_check,
            macro_df=macro_df_for_check,
            master_df=master_df_for_check,
            previous_master_path=previous_master_path,
        )
    except Exception as e:
        logger.error(f"Data quality checks failed to run: {e}")

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()