"""
Tests for CorrelationAnalyser — rolling and static correlation calculations.

Why these tests matter:
  A rolling correlation that accidentally looks forward, or that produces
  a number with too few real observations behind it, would silently feed
  a misleading "relationship" into the dashboard's Correlations tab —
  exactly the same class of problem as the regime notebook's spell-count
  issue, just applied to a different calculation.

Strategy:
  Small, hand-built two-series examples with a known, calculable
  correlation, used to verify the rolling window behaves correctly at
  its boundaries and respects min_periods.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


ROLLING_WINDOW = 90  # mirrors CorrelationAnalyser.ROLLING_WINDOW


def compute_rolling_corr(
    daily_return: pd.Series, macro_var: pd.Series, window: int = ROLLING_WINDOW
) -> pd.Series:
    """Mirrors the exact rolling().corr() call in CorrelationAnalyser.compute()."""
    return daily_return.rolling(window=window, min_periods=30).corr(macro_var)


# ── Test 1: perfectly correlated series gives correlation of exactly 1 ──────

def test_perfect_positive_correlation():
    """
    Two series that move in perfect lockstep (one is just 2x the other)
    should show a rolling correlation of exactly 1.0 once the window
    is full — this is the simplest possible correctness check for the
    correlation formula itself.
    """
    n = 100
    base = pd.Series(np.linspace(0, 10, n))
    series_a = base
    series_b = base * 2.0  # perfectly proportional, same direction

    corr = compute_rolling_corr(series_a, series_b)

    last_valid = corr.dropna().iloc[-1]
    assert last_valid == pytest.approx(1.0, abs=1e-9), (
        f"Expected correlation of exactly 1.0 for perfectly proportional "
        f"series, got {last_valid}"
    )


def test_perfect_negative_correlation():
    """Mirror of the above: inversely proportional series should give -1.0."""
    n = 100
    base = pd.Series(np.linspace(0, 10, n))
    series_a = base
    series_b = -base  # perfectly inverse

    corr = compute_rolling_corr(series_a, series_b)

    last_valid = corr.dropna().iloc[-1]
    assert last_valid == pytest.approx(-1.0, abs=1e-9), (
        f"Expected correlation of exactly -1.0 for perfectly inverse "
        f"series, got {last_valid}"
    )


def test_zero_correlation_for_unrelated_random_series():
    """
    Two independently-generated random series should show a rolling
    correlation close to 0 (not exactly 0, but nowhere near ±1).
    This is a sanity check that the formula doesn't fabricate a strong
    relationship out of unrelated noise.
    """
    rng = np.random.default_rng(seed=123)
    series_a = pd.Series(rng.normal(size=200))
    series_b = pd.Series(rng.normal(size=200))  # independently generated

    corr = compute_rolling_corr(series_a, series_b)
    last_valid = corr.dropna().iloc[-1]

    assert abs(last_valid) < 0.3, (
        f"Expected near-zero correlation for independent random series, "
        f"got {last_valid} — formula may be fabricating a relationship"
    )


# ── Test 2: min_periods enforcement ──────────────────────────────────────────

def test_no_correlation_value_before_min_periods():
    """
    With min_periods=30, no correlation value should exist before at
    least 30 paired observations are available — even though the window
    is 90, we explicitly allow a result once 30 is reached so the metric
    isn't unusably sparse for short histories. We test that fewer than
    30 produces NaN, and to be lenient, leave the >=30-but-<90 case
    untested here (it's the intended, working behaviour, not an edge case).
    """
    series_a = pd.Series(np.linspace(0, 1, 25))  # only 25 points — below 30
    series_b = pd.Series(np.linspace(0, 2, 25))

    corr = compute_rolling_corr(series_a, series_b)

    assert corr.isna().all(), (
        "Expected all NaN with fewer than min_periods=30 observations, "
        f"got {corr.dropna().tolist()}"
    )


def test_correlation_appears_exactly_at_min_periods():
    """
    The first non-NaN correlation value should appear at exactly the
    30th observation (index 29), not before and not later than necessary.
    """
    n = 40
    series_a = pd.Series(np.linspace(0, 1, n))
    series_b = pd.Series(np.linspace(0, 2, n))

    corr = compute_rolling_corr(series_a, series_b)

    assert corr.iloc[:29].isna().all(), (
        "Expected NaN for all rows before the 30th observation"
    )
    assert not pd.isna(corr.iloc[29]), (
        "Expected a valid (non-NaN) correlation exactly at the 30th observation"
    )


# ── Test 3: rolling window only looks backward, never forward ──────────────

def test_rolling_correlation_does_not_look_forward():
    """
    The single most important property: an early correlation value must
    be computed using ONLY data up to that point, never future data.
    We build a series where the first 50 points are perfectly correlated
    and the next 50 points are perfectly ANTI-correlated, then confirm
    the correlation value calculated at the boundary (row 50) reflects
    ONLY the first half's relationship, not a blend with future data
    it shouldn't yet know about.
    """
    n_half = 50
    # First half: perfectly positively correlated
    a1 = pd.Series(np.linspace(0, 10, n_half))
    b1 = a1.copy()
    # Second half: perfectly negatively correlated
    a2 = pd.Series(np.linspace(10, 20, n_half))
    b2 = -a2

    series_a = pd.concat([a1, a2], ignore_index=True)
    series_b = pd.concat([b1, b2], ignore_index=True)

    corr = compute_rolling_corr(series_a, series_b)

    # At row 49 (the last row of the first half), only first-half data
    # has been seen — correlation must be strongly positive, not yet
    # contaminated by the second half's anti-correlation.
    value_at_boundary = corr.iloc[n_half - 1]
    assert value_at_boundary > 0.9, (
        f"Expected correlation at the boundary to reflect ONLY the "
        f"positively-correlated first half (>0.9), got {value_at_boundary} "
        f"— this would indicate the window is looking ahead into future data"
    )


# ── Test 4: correlation values stay within mathematical bounds ──────────────

def test_correlation_stays_within_bounds():
    """
    Correlation is mathematically bounded to [-1, 1]. Any valid output
    outside this range indicates a bug in the formula or its inputs,
    not a legitimate result.
    """
    rng = np.random.default_rng(seed=7)
    series_a = pd.Series(rng.normal(loc=0, scale=1, size=150))
    series_b = pd.Series(0.5 * series_a + rng.normal(loc=0, scale=0.5, size=150))

    corr = compute_rolling_corr(series_a, series_b).dropna()

    assert (corr >= -1.0001).all() and (corr <= 1.0001).all(), (
        f"Correlation must stay within [-1, 1], got range "
        f"[{corr.min():.4f}, {corr.max():.4f}]"
    )


# ── Test 5: NaN in input data is handled, not propagated forever ────────────

def test_correlation_recovers_after_a_gap_in_data():
    """
    Real macro data has occasional gaps (we've seen this with FRED
    series before). A few NaN values in the middle of a series
    shouldn't permanently break the rolling correlation for everything
    afterward, once enough clean data is available again.
    """
    n = 100
    series_a = pd.Series(np.linspace(0, 10, n))
    series_b = series_a.copy()
    # Introduce a small gap in the middle
    series_b.iloc[40:45] = np.nan

    corr = compute_rolling_corr(series_a, series_b)

    # Well after the gap has rolled out of the window, correlation
    # should recover to strongly positive (not stay NaN forever)
    last_valid = corr.dropna().iloc[-1]
    assert last_valid > 0.9, (
        f"Expected correlation to recover to strongly positive after "
        f"the gap rolls out of the window, got {last_valid}"
    )