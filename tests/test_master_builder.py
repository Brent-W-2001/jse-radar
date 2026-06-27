"""
Tests for MasterBuilder — the asof merge that joins equity and macro data.

Why this file matters most:
  The asof merge is the single point in the pipeline where a subtle bug
  could introduce look-ahead bias — letting a stock "know" about a macro
  reading before it actually happened. That would silently make every
  signal and correlation downstream look better than it really is.

  These tests use small, hand-built DataFrames where we know the exact
  correct answer, rather than the real ~80,000 row dataset — this lets
  us verify exact values rather than just "it ran without crashing".

Strategy:
  We don't call MasterBuilder.build() directly, because that method reads
  parquet files from disk by date-glob, which is awkward to control in a
  test. Instead we test the core merge logic in isolation by replicating
  exactly what build() does to two small DataFrames we construct here.
  This tests the logic, not the file I/O — file I/O is simple enough
  (read parquet, write parquet) that it doesn't need dedicated tests.
"""

from __future__ import annotations

import pandas as pd
import pytest


def asof_merge_equity_macro(equity: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """
    Replicates the core merge logic from MasterBuilder.build(), extracted
    here so we can test it directly against small, known DataFrames.

    This mirrors the real implementation exactly: cast both date columns
    to datetime64[us], sort, then merge_asof with direction="backward".
    """
    equity = equity.copy()
    macro  = macro.copy()

    equity["date"] = pd.to_datetime(equity["date"]).astype("datetime64[us]")
    macro["date"]  = pd.to_datetime(macro["date"]).astype("datetime64[us]")

    equity = equity.sort_values("date").reset_index(drop=True)
    macro  = macro.sort_values("date").reset_index(drop=True)

    macro_cols = [c for c in macro.columns if c != "date"]

    return pd.merge_asof(
        equity,
        macro[["date"] + macro_cols],
        on="date",
        direction="backward",
    )


# ── Fixtures: small, known datasets ──────────────────────────────────────────

@pytest.fixture
def simple_equity() -> pd.DataFrame:
    """Five consecutive trading days for one ticker."""
    return pd.DataFrame({
        "date":   pd.to_datetime([
            "2024-01-15", "2024-01-16", "2024-01-17", "2024-01-18", "2024-01-19",
        ]),
        "ticker": ["TEST.JO"] * 5,
        "close":  [100.0, 101.0, 102.0, 103.0, 104.0],
    })


@pytest.fixture
def simple_macro() -> pd.DataFrame:
    """
    Monthly macro readings — note the gap: 31 Dec, then nothing until
    31 Jan. Every equity date in January should pick up the 31 Dec
    reading, since no January reading exists yet on those dates.
    """
    return pd.DataFrame({
        "date":      pd.to_datetime(["2023-12-31", "2024-01-31", "2024-02-29"]),
        "repo_rate": [8.25, 8.25, 8.00],
    })


# ── Test 1: basic correctness ─────────────────────────────────────────────────

def test_equity_dates_get_most_recent_prior_macro_reading(simple_equity, simple_macro):
    """
    Every equity date in mid-January 2024 falls between the 31 Dec and
    31 Jan macro readings. Since 31 Jan hasn't happened yet relative to
    these dates, every row should pick up the 31 Dec reading (8.25),
    not the 31 Jan reading.
    """
    result = asof_merge_equity_macro(simple_equity, simple_macro)

    assert len(result) == 5
    assert (result["repo_rate"] == 8.25).all(), (
        "Expected all January equity rows to use the 31 Dec macro reading, "
        f"got: {result['repo_rate'].tolist()}"
    )


# ── Test 2: no look-ahead bias ────────────────────────────────────────────────

def test_no_look_ahead_bias(simple_equity, simple_macro):
    """
    The most critical correctness property: an equity row must NEVER
    be matched to a macro reading whose date is in the future relative
    to that equity row. This directly tests for look-ahead bias.
    """
    result = asof_merge_equity_macro(simple_equity, simple_macro)

    # Re-attach the macro date so we can compare it to the equity date.
    # We do this by re-running the merge but keeping macro's date too,
    # renamed to avoid collision with equity's own date column.
    macro_renamed = simple_macro.rename(columns={"date": "macro_date"})
    macro_renamed["date"] = simple_macro["date"]  # keep original for merge key

    equity_sorted = simple_equity.copy()
    equity_sorted["date"] = pd.to_datetime(equity_sorted["date"]).astype("datetime64[us]")
    macro_renamed["date"] = pd.to_datetime(macro_renamed["date"]).astype("datetime64[us]")

    check = pd.merge_asof(
        equity_sorted.sort_values("date"),
        macro_renamed[["date", "macro_date", "repo_rate"]].sort_values("date"),
        on="date",
        direction="backward",
    )

    # The core assertion: matched macro_date must always be <= equity date
    violations = check[check["macro_date"] > check["date"]]
    assert violations.empty, (
        f"Look-ahead bias detected — these rows matched a future macro "
        f"reading:\n{violations}"
    )


# ── Test 3: a new macro reading "switches on" exactly on its date ───────────

def test_new_macro_reading_applies_from_its_own_date_onward():
    """
    If a macro reading is published on, say, 31 Jan, then an equity row
    dated exactly 31 Jan should pick up that NEW reading (not the
    previous one) — direction="backward" includes the exact match date.
    """
    equity = pd.DataFrame({
        "date":   pd.to_datetime(["2024-01-30", "2024-01-31", "2024-02-01"]),
        "ticker": ["TEST.JO"] * 3,
        "close":  [100.0, 101.0, 102.0],
    })
    macro = pd.DataFrame({
        "date":      pd.to_datetime(["2023-12-31", "2024-01-31"]),
        "repo_rate": [8.25, 8.00],
    })

    result = asof_merge_equity_macro(equity, macro)

    expected = [8.25, 8.00, 8.00]  # 30 Jan: old rate. 31 Jan onward: new rate.
    assert result["repo_rate"].tolist() == expected, (
        f"Expected {expected}, got {result['repo_rate'].tolist()}"
    )


# ── Test 4: row count is preserved ────────────────────────────────────────────

def test_merge_preserves_equity_row_count(simple_equity, simple_macro):
    """
    An asof merge should never add or drop equity rows — it only ever
    attaches columns. If this fails, something is duplicating or
    dropping rows during the merge, which would corrupt the dataset.
    """
    result = asof_merge_equity_macro(simple_equity, simple_macro)
    assert len(result) == len(simple_equity)


# ── Test 5: equity dates before ANY macro data get NaN, not an error ────────

def test_equity_dates_before_first_macro_reading_get_nan():
    """
    If an equity row is dated before any macro reading exists yet,
    there's nothing to backward-match to. This should produce NaN
    for the macro columns, not raise an exception or silently
    grab a future value.
    """
    equity = pd.DataFrame({
        "date":   pd.to_datetime(["2020-01-01"]),
        "ticker": ["TEST.JO"],
        "close":  [50.0],
    })
    macro = pd.DataFrame({
        "date":      pd.to_datetime(["2023-12-31"]),
        "repo_rate": [8.25],
    })

    result = asof_merge_equity_macro(equity, macro)

    assert len(result) == 1
    assert pd.isna(result["repo_rate"].iloc[0]), (
        "Expected NaN for an equity date before any macro data exists, "
        f"got: {result['repo_rate'].iloc[0]}"
    )


# ── Test 6: datetime precision mismatch is handled correctly ────────────────

def test_handles_mismatched_datetime_precision():
    """
    This directly tests the bug we hit in production: pyarrow can write
    timestamps as datetime64[ms] in one file and datetime64[us] in
    another. merge_asof raises an error if the two merge keys have
    different precision. Our function casts both to [us] before merging
    — this test simulates that exact mismatch to confirm the fix holds.
    """
    equity = pd.DataFrame({
        "date":   pd.array(
            pd.to_datetime(["2024-01-15", "2024-01-16"]), dtype="datetime64[ms]"
        ),
        "ticker": ["TEST.JO"] * 2,
        "close":  [100.0, 101.0],
    })
    macro = pd.DataFrame({
        "date":      pd.array(
            pd.to_datetime(["2024-01-01"]), dtype="datetime64[ns]"
        ),
        "repo_rate": [8.25],
    })

    # Should not raise — this would have thrown
    # "incompatible merge keys" before the fix
    result = asof_merge_equity_macro(equity, macro)
    assert len(result) == 2
    assert (result["repo_rate"] == 8.25).all()


# ── Test 7: multiple tickers don't interfere with each other ────────────────

def test_multiple_tickers_each_get_correct_macro_value():
    """
    With multiple tickers on the same dates, every ticker on a given
    date should receive the SAME macro reading — the macro merge
    shouldn't behave differently depending on which ticker a row
    belongs to (macro data has no concept of ticker).
    """
    equity = pd.DataFrame({
        "date":   pd.to_datetime(
            ["2024-01-15", "2024-01-15", "2024-01-20", "2024-01-20"]
        ),
        "ticker": ["AAA.JO", "BBB.JO", "AAA.JO", "BBB.JO"],
        "close":  [10.0, 20.0, 11.0, 21.0],
    })
    macro = pd.DataFrame({
        "date":      pd.to_datetime(["2024-01-01", "2024-01-18"]),
        "repo_rate": [8.25, 8.00],
    })

    result = asof_merge_equity_macro(equity, macro)

    jan_15_rates = result[result["date"] == "2024-01-15"]["repo_rate"]
    jan_20_rates = result[result["date"] == "2024-01-20"]["repo_rate"]

    assert (jan_15_rates == 8.25).all(), "Both tickers on 15 Jan should get the same rate"
    assert (jan_20_rates == 8.00).all(), "Both tickers on 20 Jan should get the same rate"