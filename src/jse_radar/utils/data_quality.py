"""
Data quality checks — runs as the final step of the pipeline.

Why this exists:
  Over the course of building this project, three real bugs slipped
  through silently and were only caught by manually reading log output
  carefully: a FRED series being retired (returned empty, not an error),
  wbgapi changing its return type (returned a dict, not a DataFrame),
  and an RSI edge case (returned NaN forever for a stock with no losing
  days, with no obvious symptom other than a blank dashboard cell).

  None of these crashed the pipeline — they all completed "successfully"
  while quietly producing wrong or incomplete data. This module exists
  to catch the NEXT one of these automatically, by checking known
  properties that should always hold if the pipeline ran correctly,
  rather than relying on someone noticing a gap in a chart.

Design principle:
  This module NEVER raises an exception or stops the pipeline. A data
  quality issue means something needs attention, not that everything
  else that succeeded should be thrown away too. Every check logs a
  clear WARNING (or ERROR for the most severe issues) and the pipeline
  continues. The person running it sees these in the log and in
  scheduler.log, and can investigate without losing partial progress.

What we check:
  1. Missing equity trading days — any ticker with an unusually high
     proportion of missing/NaN close prices over the fetch window
  2. Frozen/zero-variance price feeds — a ticker whose price hasn't
     moved at all over a long window (suggests a stale or broken feed,
     not a genuinely flat stock)
  3. Macro series completeness — any FRED/World Bank series that came
     back with suspiciously few observations
  4. Master frame row count — an unexpected drop in row count compared
     to the previous run, which would indicate something upstream
     silently lost data
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field

from jse_radar.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class QualityIssue:
    """A single data quality finding."""
    severity: str   # "WARNING" or "ERROR"
    check:    str   # which check raised this
    message:  str


@dataclass
class QualityReport:
    """Collects every issue found during a data quality run."""
    issues: list[QualityIssue] = field(default_factory=list)

    def add(self, severity: str, check: str, message: str) -> None:
        self.issues.append(QualityIssue(severity, check, message))

    @property
    def has_errors(self) -> bool:
        return any(i.severity == "ERROR" for i in self.issues)

    def log_summary(self) -> None:
        if not self.issues:
            logger.info("Data quality check: no issues found.")
            return

        errors   = [i for i in self.issues if i.severity == "ERROR"]
        warnings = [i for i in self.issues if i.severity == "WARNING"]

        logger.info(
            f"Data quality check: {len(errors)} error(s), "
            f"{len(warnings)} warning(s) found."
        )
        for issue in self.issues:
            log_fn = logger.error if issue.severity == "ERROR" else logger.warning
            log_fn(f"  [{issue.check}] {issue.message}")


# ── Check 1: missing equity trading days ─────────────────────────────────────

def check_missing_trading_days(
    equity_df: pd.DataFrame, report: QualityReport, max_missing_pct: float = 5.0
) -> None:
    """
    Flag any ticker where more than `max_missing_pct`% of its expected
    trading days have a NaN close price (after forward-fill has already
    been applied). A high missing rate suggests the fetch genuinely
    failed for that ticker, or it was delisted partway through and
    nobody removed it from the universe.
    """
    if "ticker" not in equity_df.columns or "close" not in equity_df.columns:
        report.add("WARNING", "missing_trading_days",
                    "Equity frame missing 'ticker' or 'close' column — skipped check")
        return

    missing_pct = (
        equity_df.groupby("ticker")["close"]
        .apply(lambda s: s.isna().mean() * 100)
    )

    flagged = missing_pct[missing_pct > max_missing_pct]
    for ticker, pct in flagged.items():
        report.add(
            "WARNING", "missing_trading_days",
            f"{ticker}: {pct:.1f}% of close prices are missing "
            f"(threshold: {max_missing_pct}%) — check for delisting or fetch failure"
        )


# ── Check 2: frozen / zero-variance price feeds ──────────────────────────────

def check_frozen_price_feeds(
    equity_df: pd.DataFrame, report: QualityReport, min_window: int = 30
) -> None:
    """
    Flag any ticker whose closing price hasn't changed AT ALL over its
    most recent `min_window` trading days. A real stock essentially
    never trades perfectly flat for 30 consecutive days — this almost
    always indicates a frozen/stale data feed rather than genuine
    zero volatility.
    """
    if "ticker" not in equity_df.columns or "close" not in equity_df.columns:
        return

    for ticker, group in equity_df.groupby("ticker"):
        recent = group.sort_values("date")["close"].tail(min_window).dropna()
        if len(recent) < min_window:
            continue  # not enough recent history to judge — skip rather than guess
        if recent.nunique() == 1:
            report.add(
                "WARNING", "frozen_price_feed",
                f"{ticker}: close price has not changed at all over the "
                f"last {min_window} trading days — possible stale data feed"
            )


# ── Check 3: macro series completeness ────────────────────────────────────────

def check_macro_completeness(
    macro_df: pd.DataFrame, report: QualityReport, min_observations: int = 10
) -> None:
    """
    Flag any macro column that came back with fewer than
    `min_observations` non-null values. This is exactly the symptom
    we hit with the retired FRED repo_rate series — it "succeeded"
    but returned zero usable observations.
    """
    if macro_df is None or macro_df.empty:
        report.add("ERROR", "macro_completeness",
                    "Macro DataFrame is empty or None")
        return

    for col in macro_df.columns:
        if col == "date":
            continue
        non_null = macro_df[col].notna().sum()
        if non_null < min_observations:
            report.add(
                "ERROR" if non_null == 0 else "WARNING",
                "macro_completeness",
                f"'{col}': only {non_null} non-null observation(s) "
                f"(threshold: {min_observations}) — series may be retired, "
                f"renamed, or the fetch silently failed"
            )


# ── Check 4: master frame row count regression ───────────────────────────────

def check_row_count_regression(
    current_df: pd.DataFrame,
    previous_path: Path | None,
    report: QualityReport,
    max_drop_pct: float = 10.0,
) -> None:
    """
    Compare the current master frame's row count against the previous
    run's saved file. A significant unexpected DROP suggests something
    upstream silently lost data (e.g. a fetch partially failed but
    didn't raise an error, or a join condition changed and is now
    excluding rows that used to match).

    A drop is expected and fine the very first time this runs (no
    previous file to compare against) or when --start is changed to a
    later date deliberately — this check is a prompt to investigate,
    not a guarantee something is wrong.
    """
    if previous_path is None or not previous_path.exists():
        logger.debug("No previous master frame found — skipping row count comparison")
        return

    try:
        previous_df = pd.read_parquet(previous_path, engine="pyarrow")
    except Exception as e:
        report.add("WARNING", "row_count_regression",
                    f"Could not read previous master frame for comparison: {e}")
        return

    current_rows  = len(current_df)
    previous_rows = len(previous_df)

    if previous_rows == 0:
        return

    drop_pct = (previous_rows - current_rows) / previous_rows * 100

    if drop_pct > max_drop_pct:
        report.add(
            "ERROR", "row_count_regression",
            f"Master frame row count dropped {drop_pct:.1f}% "
            f"({previous_rows:,} -> {current_rows:,} rows) versus the previous "
            f"run — investigate before trusting this run's output"
        )


# ── Orchestration ─────────────────────────────────────────────────────────────

def run_all_checks(
    equity_df: pd.DataFrame | None = None,
    macro_df:  pd.DataFrame | None = None,
    master_df: pd.DataFrame | None = None,
    previous_master_path: Path | None = None,
) -> QualityReport:
    """
    Run every available check against whichever DataFrames were passed
    in. Each argument is optional — the pipeline calls this once per
    stage with whatever it has at that point, so a single missing
    input doesn't prevent the other checks from running.
    """
    report = QualityReport()

    if equity_df is not None and not equity_df.empty:
        check_missing_trading_days(equity_df, report)
        check_frozen_price_feeds(equity_df, report)

    if macro_df is not None:
        check_macro_completeness(macro_df, report)

    if master_df is not None and not master_df.empty:
        check_row_count_regression(master_df, previous_master_path, report)

    report.log_summary()
    return report