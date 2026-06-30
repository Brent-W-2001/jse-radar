"""
One-off script: run the full walk-forward backtest on real data and
print the honest summary.

Run from the repo root with the jse-radar conda environment active:
    conda activate jse-radar
    python scripts/run_backtest.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from jse_radar.config import PROC_MASTER_DIR
from jse_radar.modeling.regime_predictor import build_outperformance_label
from jse_radar.modeling.backtester import run_walk_forward_backtest, summarise_backtest
from jse_radar.utils.logger import get_logger

logger = get_logger("run_backtest")


def main():
    master_files = sorted(PROC_MASTER_DIR.glob("master_regimes_*.parquet"))
    if not master_files:
        raise FileNotFoundError(
            "No master_regimes parquet found. Run scripts/run_pipeline.py first."
        )

    logger.info(f"Loading {master_files[-1]}")
    df = pd.read_parquet(master_files[-1], engine="pyarrow")
    df["date"] = pd.to_datetime(df["date"])

    logger.info("Building outperformance labels...")
    df = build_outperformance_label(df)

    logger.info("Running walk-forward backtest (this may take a few minutes — "
                "the model refits at every monthly rebalance date)...")
    results = run_walk_forward_backtest(df)

    if results.empty:
        logger.warning("Backtest produced no results — check the logs above for why.")
        return

    summary = summarise_backtest(results)

    print("\n" + "=" * 70)
    print("WALK-FORWARD BACKTEST SUMMARY")
    print("=" * 70)
    for key, value in summary.items():
        print(f"{key:30s}: {value}")
    print("=" * 70)

    print("\nPer-period results (most recent 10):")
    print(
        results[["rebalance_date", "picked_tickers", "portfolio_return",
                  "alsi_return", "excess_return", "beat_alsi"]]
        .tail(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()