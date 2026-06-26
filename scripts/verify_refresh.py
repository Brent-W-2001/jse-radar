"""
Verify the last pipeline refresh — checks data freshness.

Run this any time you want to confirm your data is up to date:
    conda activate jse-radar
    python scripts/verify_refresh.py

Prints a summary of the most recent data in each parquet file
so you can confirm the scheduled refresh is working correctly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from datetime import date, timedelta
from jse_radar.config import (
    RAW_EQUITY_DIR, RAW_MACRO_DIR,
    PROC_EQUITY_DIR, PROC_MACRO_DIR, PROC_MASTER_DIR,
)


def check_dir(label: str, directory: Path, pattern: str) -> None:
    """Find the latest file matching pattern and report its freshness."""
    files = sorted(directory.glob(pattern), key=lambda f: f.stat().st_mtime)
    if not files:
        print(f"  ❌  {label}: no files found in {directory}")
        return

    latest = files[-1]
    modified = date.fromtimestamp(latest.stat().st_mtime)
    age_days = (date.today() - modified).days

    # Try to read the max date from the file itself
    try:
        df = pd.read_parquet(latest, engine="pyarrow")
        if "date" in df.columns:
            max_date = pd.to_datetime(df["date"]).max().date()
            rows     = len(df)
            status   = "✅" if age_days <= 3 else "⚠️ "
            print(
                f"  {status}  {label:<30} "
                f"file: {latest.name}  |  "
                f"latest data: {max_date}  |  "
                f"rows: {rows:,}  |  "
                f"file age: {age_days}d"
            )
        elif "year" in df.columns:
            max_year = df["year"].max()
            print(
                f"  ✅  {label:<30} "
                f"file: {latest.name}  |  "
                f"latest year: {max_year}  |  "
                f"file age: {age_days}d"
            )
        else:
            print(
                f"  ✅  {label:<30} "
                f"file: {latest.name}  |  "
                f"file age: {age_days}d"
            )
    except Exception as e:
        print(f"  ⚠️   {label}: could not read file — {e}")


def main():
    print()
    print("jse-radar data freshness check")
    print("=" * 70)
    print()
    print("RAW DATA:")
    check_dir("Equity prices",      RAW_EQUITY_DIR,  "equities_*.parquet")
    check_dir("Macro (FRED)",        RAW_MACRO_DIR,   "macro_*.parquet")

    print()
    print("PROCESSED DATA:")
    check_dir("Equity processed",   PROC_EQUITY_DIR, "equities_processed_*.parquet")
    check_dir("Macro processed",     PROC_MACRO_DIR,  "macro_processed_*.parquet")

    print()
    print("MASTER FRAMES:")
    check_dir("Master (base)",       PROC_MASTER_DIR, "master_2*.parquet")
    check_dir("Master + signals",    PROC_MASTER_DIR, "master_signals_*.parquet")
    check_dir("Master + regimes",    PROC_MASTER_DIR, "master_regimes_*.parquet")
    check_dir("Rolling correlations",PROC_MASTER_DIR, "rolling_correlations_*.parquet")

    print()
    print("=" * 70)
    print("⚠️  = file is more than 3 days old — consider re-running the pipeline")
    print()


if __name__ == "__main__":
    main()