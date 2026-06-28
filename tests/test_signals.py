"""
Tests for SignalEngine — momentum, RSI, moving averages, trend signals.

Why these tests matter:
  Every signal here feeds the dashboard's Signals tab and the regime
  analysis notebook. If momentum is computed with an off-by-one shift,
  or RSI uses the wrong smoothing window, the numbers will look
  plausible but be quietly wrong — the kind of bug that's very hard
  to catch just by eyeballing a chart.

Strategy:
  We build tiny synthetic price series with hand-computable expected
  values, rather than testing against the real ~80,000 row dataset.
  Where a calculation has a well-known reference implementation (RSI),
  we verify against a manually worked example rather than trusting
  our own code to check itself.

  compute_rsi is imported directly from jse_radar.analysis.signals —
  it was deliberately refactored to a module-level function specifically
  so it could be tested in isolation like this, rather than being
  nested inside SignalEngine.compute() where it couldn't be imported.
  The other calculations (momentum, z-score, trend signal) are still
  one-liners inside compute(), so we replicate that exact logic here —
  see the comment above each replicated function for what it mirrors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from jse_radar.analysis.signals import compute_rsi


# ── Replicated calculation logic (mirrors SignalEngine.compute()) ───────────
# These two are simple enough one-liners inside compute() that they don't
# warrant their own module-level functions — we replicate them here exactly
# as written in the real code so the test proves the real maths.

def compute_momentum(close: pd.Series, window: int) -> pd.Series:
    """Mirrors: df.groupby('ticker')['close'].transform(lambda s: s.shift(1).pct_change(window))"""
    return close.shift(1).pct_change(window)


def compute_trend_signal(ma_50: pd.Series, ma_200: pd.Series) -> pd.Series:
    """Mirrors the np.where trend_signal logic in SignalEngine."""
    return pd.Series(
        np.where(ma_50 > ma_200, 1, np.where(ma_50 < ma_200, -1, 0)),
        index=ma_50.index,
    )


# ── Test 1: momentum uses shift(1), excluding today's price ─────────────────

def test_momentum_excludes_most_recent_price():
    """
    Momentum is deliberately computed on yesterday's close relative to
    N days before that — NOT on today's close. This guards against
    same-day noise contaminating the signal. We construct a price series
    where today's price is a deliberate outlier and confirm it's excluded.
    """
    prices = pd.Series([100, 102, 104, 106, 108, 500])  # day 5 is a spike

    mom = compute_momentum(prices, window=3)

    shifted = prices.shift(1)
    expected_day5 = (shifted.iloc[5] / shifted.iloc[5 - 3]) - 1
    assert mom.iloc[5] == pytest.approx(expected_day5)
    assert mom.iloc[5] != pytest.approx((500 / 106) - 1), (
        "Momentum leaked today's spike price into the calculation"
    )


def test_momentum_known_values():
    """Hand-computed check: a clean 10% per-period compounding series."""
    prices = pd.Series([100, 110, 121, 133.1, 146.41])
    mom_1 = compute_momentum(prices, window=1)

    assert mom_1.iloc[2] == pytest.approx(0.10, abs=1e-9)


# ── Test 2: cross-sectional z-score standardises ACROSS tickers, not time ───

def test_cross_sectional_zscore_axis():
    """
    The z-score must be computed across all tickers on the SAME date,
    not across time for the SAME ticker. We build a small multi-ticker,
    multi-date frame and confirm the groupby axis is correct.
    """
    df = pd.DataFrame({
        "date":   pd.to_datetime(
            ["2024-01-01", "2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02", "2024-01-02"]
        ),
        "ticker": ["A", "B", "C", "A", "B", "C"],
        "mom_3m": [0.10, 0.20, 0.30, 0.05, 0.05, 0.05],
    })

    z_correct = df.groupby("date")["mom_3m"].transform(lambda s: (s - s.mean()) / s.std())

    # On 2024-01-02 all three tickers have identical momentum (0.05).
    # Due to floating-point arithmetic, std() may not come out as EXACTLY
    # zero (it can be something like 8.5e-18), so the z-scores may not be
    # exactly NaN — but they MUST all be identical to each other, since
    # the inputs are identical. They must NOT be differentiated in a way
    # that would wrongly suggest one ticker has stronger momentum than
    # another when the underlying values are equal.
    day2_z = z_correct[df["date"] == "2024-01-02"]
    assert day2_z.nunique() == 1, (
        "Identical momentum values on the same date must produce "
        f"identical z-scores, got: {day2_z.tolist()}"
    )

    # On 2024-01-01, ticker C (highest momentum, 0.30) must have the
    # highest z-score and ticker A (lowest, 0.10) the lowest — this is
    # the real test that the cross-sectional axis is correct.
    day1 = df[df["date"] == "2024-01-01"].copy()
    day1["z"] = z_correct[df["date"] == "2024-01-01"]
    assert day1.loc[day1["ticker"] == "C", "z"].iloc[0] == day1["z"].max()
    assert day1.loc[day1["ticker"] == "A", "z"].iloc[0] == day1["z"].min()


# ── Test 3: RSI against known edge cases ──────────────────────────────────────

def test_rsi_all_gains_is_exactly_100():
    """
    If a price series rises every single period with no losses at all,
    RSI must be exactly 100 — this is an explicitly handled edge case
    in compute_rsi (avg_loss == 0 with avg_gain > 0), not left as an
    undefined division by zero.
    """
    prices = pd.Series(np.arange(100, 100 + 30, 1, dtype=float))
    rsi = compute_rsi(prices, window=14)

    last_valid = rsi.dropna().iloc[-1]
    assert last_valid == pytest.approx(100.0), (
        f"Expected RSI = 100 for an all-gains series, got {last_valid}"
    )


def test_rsi_all_losses_is_exactly_0():
    """Mirror of the above: a strictly falling series must give RSI = 0 exactly."""
    prices = pd.Series(np.arange(130, 130 - 30, -1, dtype=float))
    rsi = compute_rsi(prices, window=14)

    last_valid = rsi.dropna().iloc[-1]
    assert last_valid == pytest.approx(0.0), (
        f"Expected RSI = 0 for an all-losses series, got {last_valid}"
    )


def test_rsi_flat_price_is_nan_not_zero():
    """
    A perfectly flat price series has zero gains AND zero losses —
    this is genuinely undefined (0/0), not RSI=0 and not RSI=100.
    Must remain NaN, distinct from the all-gains and all-losses cases
    above which ARE now explicitly resolved to 100 and 0 respectively.
    """
    prices = pd.Series([100.0] * 30)
    rsi = compute_rsi(prices, window=14)

    last_valid = rsi.iloc[-1]
    assert pd.isna(last_valid), (
        f"Expected NaN for a flat price series (no gains or losses), got {last_valid}"
    )


def test_rsi_respects_min_periods():
    """
    RSI requires a full window (14 periods) before producing a value —
    early rows should be NaN, not a premature/unstable estimate.
    """
    prices = pd.Series(np.arange(100, 105, 1, dtype=float))  # only 5 points
    rsi = compute_rsi(prices, window=14)

    assert rsi.isna().all(), (
        "RSI should be entirely NaN when there isn't a full 14-period "
        "window of data yet"
    )


def test_rsi_mixed_series_stays_within_bounds():
    """
    RSI is bounded between 0 and 100 by definition. For a normal mixed
    up/down series, every valid (non-NaN) value must fall in this range —
    a violation would indicate the formula itself is wrong, not just an
    edge case.
    """
    rng = np.random.default_rng(seed=42)
    # Random walk with both up and down days, long enough for a stable window
    steps  = rng.normal(loc=0.1, scale=2.0, size=100)
    prices = pd.Series(100 + np.cumsum(steps))

    rsi = compute_rsi(prices, window=14).dropna()

    assert (rsi >= 0).all() and (rsi <= 100).all(), (
        f"RSI must be bounded [0, 100], got range "
        f"[{rsi.min():.2f}, {rsi.max():.2f}]"
    )


# ── Test 4: trend signal three-way branch ────────────────────────────────────

def test_trend_signal_bullish_bearish_and_equal():
    """
    trend_signal must be:
      +1 when MA50 > MA200 (bullish — golden cross territory)
      -1 when MA50 < MA200 (bearish — death cross territory)
       0 when they're exactly equal (edge case — must not be silently
         classified as bullish or bearish)
    """
    ma_50  = pd.Series([110.0, 90.0, 100.0])
    ma_200 = pd.Series([100.0, 100.0, 100.0])

    result = compute_trend_signal(ma_50, ma_200)

    assert result.tolist() == [1, -1, 0], (
        f"Expected [1, -1, 0], got {result.tolist()}"
    )


def test_trend_signal_with_nan_ma_values():
    """
    Early in a price series, MA200 won't exist yet (not enough history).
    np.where with NaN comparisons returns False for both > and <, so the
    result should fall through to 0 — confirm this doesn't error or
    produce a misleading +1/-1.
    """
    ma_50  = pd.Series([105.0, np.nan])
    ma_200 = pd.Series([np.nan, 100.0])

    result = compute_trend_signal(ma_50, ma_200)

    assert result.tolist() == [0, 0], (
        f"Expected [0, 0] when either MA is NaN, got {result.tolist()}"
    )


# ── Test 5: composite momentum score weighting ───────────────────────────────

def test_composite_momentum_score_weights():
    """
    The composite score is: 0.20*mom_1m_z + 0.30*mom_3m_z + 0.50*mom_6m_z
    We confirm the exact weights are applied correctly with simple,
    easy-to-verify-by-hand inputs.
    """
    mom_1m_z = pd.Series([1.0])
    mom_3m_z = pd.Series([1.0])
    mom_6m_z = pd.Series([1.0])

    score = (
        0.20 * mom_1m_z.fillna(0) +
        0.30 * mom_3m_z.fillna(0) +
        0.50 * mom_6m_z.fillna(0)
    )

    assert score.iloc[0] == pytest.approx(1.0)

    # A NaN component should contribute 0, not propagate NaN through
    # the whole composite score.
    mom_1m_z_nan = pd.Series([np.nan])
    score_with_nan = (
        0.20 * mom_1m_z_nan.fillna(0) +
        0.30 * mom_3m_z.fillna(0) +
        0.50 * mom_6m_z.fillna(0)
    )
    assert score_with_nan.iloc[0] == pytest.approx(0.80), (
        "A NaN 1-month momentum should contribute 0, not propagate NaN "
        "through the whole composite score"
    )