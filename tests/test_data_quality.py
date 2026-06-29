"""
Tests for the data quality checks — the final safety net that runs
after every pipeline stage completes.

Why these tests matter:
  This module exists specifically to catch silent data failures
  automatically. If the checks themselves have a bug — failing to
  fire when they should, or firing constantly on perfectly healthy
  data — the safety net is worthless. These tests prove both
  directions: each check correctly flags the problem it's designed
  to catch, AND stays silent on data that doesn't have that problem.

Strategy:
  Small, hand-built DataFrames engineered to deliberately trigger
  (or deliberately NOT trigger) each check, with clear pass/fail
  thresholds matching the real implementation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from jse_radar.utils.data_quality import (
    QualityReport,
    check_missing_trading_days,
    check_frozen_price_feeds,
    check_macro_completeness,
    check_row_count_regression,
    run_all_checks,
)


# ── Test 1: missing trading days ─────────────────────────────────────────────

def test_flags_ticker_with_high_missing_percentage():
    """
    A ticker with 50% missing close prices (well above the 5% default
    threshold) must be flagged.
    """
    df = pd.DataFrame({
        "ticker": ["BAD.JO"] * 20,
        "close":  [100.0, np.nan] * 10,  # exactly 50% missing
    })
    report = QualityReport()
    check_missing_trading_days(df, report)

    assert len(report.issues) == 1
    assert report.issues[0].check == "missing_trading_days"
    assert "BAD.JO" in report.issues[0].message


def test_does_not_flag_ticker_with_low_missing_percentage():
    """
    A ticker with only 1 missing value out of 100 (1%, well under the
    5% threshold) must NOT be flagged — this proves the check isn't
    overly sensitive and won't generate noise on perfectly normal data.
    """
    closes = [100.0] * 99 + [np.nan]  # 1% missing
    df = pd.DataFrame({
        "ticker": ["GOOD.JO"] * 100,
        "close":  closes,
    })
    report = QualityReport()
    check_missing_trading_days(df, report)

    assert len(report.issues) == 0, (
        f"Expected no issues for 1% missing data, got: {report.issues}"
    )


def test_missing_trading_days_handles_multiple_tickers_independently():
    """
    With multiple tickers, only the genuinely problematic one should
    be flagged — a bad ticker must not cause a healthy ticker to be
    flagged too, and vice versa.
    """
    df = pd.DataFrame({
        "ticker": ["BAD.JO"] * 20 + ["GOOD.JO"] * 20,
        "close":  ([100.0, np.nan] * 10) + ([100.0] * 20),
    })
    report = QualityReport()
    check_missing_trading_days(df, report)

    assert len(report.issues) == 1
    assert "BAD.JO" in report.issues[0].message
    assert "GOOD.JO" not in report.issues[0].message


# ── Test 2: frozen price feeds ────────────────────────────────────────────────

def test_flags_frozen_price_feed():
    """
    A ticker whose last 30 closes are all identical must be flagged —
    this is the textbook symptom of a stale/broken feed, not a
    genuinely flat-trading stock.
    """
    df = pd.DataFrame({
        "ticker": ["FROZEN.JO"] * 35,
        "date":   pd.date_range("2024-01-01", periods=35, freq="D"),
        "close":  [100.0] * 35,  # never moves, even over 35 days
    })
    report = QualityReport()
    check_frozen_price_feeds(df, report, min_window=30)

    assert len(report.issues) == 1
    assert report.issues[0].check == "frozen_price_feed"
    assert "FROZEN.JO" in report.issues[0].message


def test_does_not_flag_normally_trading_ticker():
    """A ticker with genuine day-to-day price movement must NOT be flagged."""
    rng = np.random.default_rng(seed=5)
    prices = 100 + np.cumsum(rng.normal(0, 1, 35))
    df = pd.DataFrame({
        "ticker": ["NORMAL.JO"] * 35,
        "date":   pd.date_range("2024-01-01", periods=35, freq="D"),
        "close":  prices,
    })
    report = QualityReport()
    check_frozen_price_feeds(df, report, min_window=30)

    assert len(report.issues) == 0, (
        f"Expected no issues for normally-trading prices, got: {report.issues}"
    )


def test_frozen_check_skips_ticker_with_insufficient_history():
    """
    A ticker with fewer than min_window days of history shouldn't be
    judged at all — there isn't enough data to distinguish "frozen"
    from "just listed yesterday".
    """
    df = pd.DataFrame({
        "ticker": ["NEW.JO"] * 10,  # only 10 days, below min_window=30
        "date":   pd.date_range("2024-01-01", periods=10, freq="D"),
        "close":  [100.0] * 10,
    })
    report = QualityReport()
    check_frozen_price_feeds(df, report, min_window=30)

    assert len(report.issues) == 0, (
        "A ticker with insufficient history should be skipped, not flagged"
    )


# ── Test 3: macro completeness ────────────────────────────────────────────────

def test_flags_macro_column_with_zero_observations_as_error():
    """
    A macro column that came back completely empty (the exact symptom
    of the retired FRED repo_rate series we hit in production) must be
    flagged as an ERROR — the most severe category, since this column
    is entirely unusable.
    """
    df = pd.DataFrame({
        "date":      pd.date_range("2024-01-01", periods=12, freq="ME"),
        "repo_rate": [np.nan] * 12,  # zero observations
        "cpi_all":   np.linspace(100, 105, 12),  # healthy column
    })
    report = QualityReport()
    check_macro_completeness(df, report, min_observations=10)

    repo_issues = [i for i in report.issues if "repo_rate" in i.message]
    assert len(repo_issues) == 1
    assert repo_issues[0].severity == "ERROR"

    cpi_issues = [i for i in report.issues if "cpi_all" in i.message]
    assert len(cpi_issues) == 0, "Healthy column should not be flagged"


def test_flags_macro_column_with_few_observations_as_warning():
    """
    A column with SOME data but below the threshold (e.g. a newly
    added series that's only just started reporting) should be a
    WARNING, not an ERROR — it's a real concern but not a total failure.
    """
    df = pd.DataFrame({
        "date":     pd.date_range("2024-01-01", periods=12, freq="ME"),
        "new_series": [1.0, 2.0, 3.0] + [np.nan] * 9,  # only 3 observations
    })
    report = QualityReport()
    check_macro_completeness(df, report, min_observations=10)

    assert len(report.issues) == 1
    assert report.issues[0].severity == "WARNING"


def test_does_not_flag_healthy_macro_dataframe():
    """A fully populated macro DataFrame should produce zero issues."""
    df = pd.DataFrame({
        "date":      pd.date_range("2024-01-01", periods=24, freq="ME"),
        "tbill_rate": np.linspace(7.0, 8.0, 24),
        "cpi_all":    np.linspace(100, 110, 24),
    })
    report = QualityReport()
    check_macro_completeness(df, report, min_observations=10)

    assert len(report.issues) == 0


def test_flags_entirely_empty_macro_dataframe():
    """An empty or None macro DataFrame is an unambiguous ERROR."""
    report = QualityReport()
    check_macro_completeness(pd.DataFrame(), report)

    assert len(report.issues) == 1
    assert report.issues[0].severity == "ERROR"


# ── Test 4: row count regression ──────────────────────────────────────────────

def test_flags_significant_row_count_drop(tmp_path: Path):
    """
    A master frame that's 20% smaller than the previous run's saved
    file (well above the 10% threshold) must be flagged as an ERROR —
    this is the exact symptom that would catch an upstream fetch or
    join silently losing data.
    """
    previous_df = pd.DataFrame({"ticker": ["A.JO"] * 1000})
    previous_path = tmp_path / "master_previous.parquet"
    previous_df.to_parquet(previous_path)

    current_df = pd.DataFrame({"ticker": ["A.JO"] * 800})  # 20% fewer rows

    report = QualityReport()
    check_row_count_regression(current_df, previous_path, report, max_drop_pct=10.0)

    assert len(report.issues) == 1
    assert report.issues[0].severity == "ERROR"
    assert report.issues[0].check == "row_count_regression"


def test_does_not_flag_small_row_count_change(tmp_path: Path):
    """
    A small change (e.g. one extra trading day fetched since the last
    run) should NOT be flagged — only a drop ABOVE the threshold matters,
    and a small increase or decrease is completely normal day to day.
    """
    previous_df = pd.DataFrame({"ticker": ["A.JO"] * 1000})
    previous_path = tmp_path / "master_previous.parquet"
    previous_df.to_parquet(previous_path)

    current_df = pd.DataFrame({"ticker": ["A.JO"] * 1005})  # slightly MORE rows

    report = QualityReport()
    check_row_count_regression(current_df, previous_path, report, max_drop_pct=10.0)

    assert len(report.issues) == 0


def test_row_count_check_handles_missing_previous_file_gracefully():
    """
    On the very first pipeline run, there's no previous file to compare
    against. This must not crash or raise — it should simply skip the
    comparison silently.
    """
    current_df = pd.DataFrame({"ticker": ["A.JO"] * 100})
    report = QualityReport()

    # Should not raise
    check_row_count_regression(current_df, None, report)
    assert len(report.issues) == 0

    # Also shouldn't raise for a path that doesn't exist on disk
    check_row_count_regression(
        current_df, Path("D:/this/path/does/not/exist.parquet"), report
    )
    assert len(report.issues) == 0


# ── Test 5: orchestration — run_all_checks never raises ─────────────────────

def test_run_all_checks_handles_all_none_gracefully():
    """
    If every input is None (e.g. an earlier pipeline phase failed
    entirely and produced nothing), run_all_checks must not crash —
    it should simply produce an empty, clean report.
    """
    report = run_all_checks(
        equity_df=None, macro_df=None, master_df=None, previous_master_path=None
    )
    assert isinstance(report, QualityReport)
    assert len(report.issues) == 0


def test_quality_report_has_errors_property():
    """
    The has_errors property must correctly distinguish a report
    containing at least one ERROR from one with only WARNINGs (or
    nothing at all) — this is what a caller would check to decide
    whether to escalate.
    """
    report = QualityReport()
    assert report.has_errors is False

    report.add("WARNING", "some_check", "a minor issue")
    assert report.has_errors is False, "Warnings alone should not count as errors"

    report.add("ERROR", "some_check", "a serious issue")
    assert report.has_errors is True