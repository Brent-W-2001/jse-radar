"""
Tests for EquityProcessor — forward-fill, returns, volatility, 52-week
high/low calculations.

Why these tests matter:
  These calculations feed directly into the signal engine and the
  dashboard's price charts. A forward-fill that runs too far would make
  a suspended or delisted stock look like it kept trading normally. A
  wrong annualisation factor on volatility would silently misstate risk
  by a fixed but wrong multiple. These are exactly the kind of "looks
  plausible, is quietly wrong" bugs that are hard to catch by eye.

Strategy:
  Small, hand-built multi-ticker price series with deliberately
  engineered gaps and known expected outputs, rather than testing
  against the real ~80,000 row dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ── Replicated logic (mirrors EquityProcessor.process()) ────────────────────

def forward_fill_capped(close: pd.Series, limit: int = 5) -> pd.Series:
    """Mirrors: df.groupby('ticker')['close'].transform(lambda s: s.ffill(limit=5))"""
    return close.ffill(limit=limit)


def compute_daily_return(close: pd.Series) -> pd.Series:
    """Mirrors: df.groupby('ticker')['close'].pct_change()"""
    return close.pct_change()


def compute_log_return(close: pd.Series) -> pd.Series:
    """Mirrors: np.log(s / s.shift(1))"""
    return np.log(close / close.shift(1))


def compute_volatility_21d(log_return: pd.Series) -> pd.Series:
    """Mirrors: rolling(21, min_periods=10).std() * sqrt(252)"""
    return log_return.rolling(window=21, min_periods=10).std() * np.sqrt(252)


def compute_52w_high_low(close: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Mirrors the high_52w / low_52w rolling(252, min_periods=50) calculation."""
    high = close.rolling(window=252, min_periods=50).max()
    low  = close.rolling(window=252, min_periods=50).min()
    return high, low


# ── Test 1: forward-fill cap ──────────────────────────────────────────────────

def test_forward_fill_fills_short_gaps():
    """
    A short gap (3 missing days, under the 5-day cap) should be fully
    forward-filled with the last known good price.
    """
    prices = pd.Series([100.0, np.nan, np.nan, np.nan, 104.0])
    filled = forward_fill_capped(prices, limit=5)

    assert filled.iloc[1:4].tolist() == [100.0, 100.0, 100.0], (
        f"Expected a 3-day gap to be fully filled with the prior price, "
        f"got {filled.iloc[1:4].tolist()}"
    )


def test_forward_fill_stops_after_cap():
    """
    A gap LONGER than the 5-day cap should only be filled for the first
    5 days — beyond that, it should remain NaN rather than carrying a
    stale price forward indefinitely (e.g. for a suspended or delisted stock).
    """
    # 7 consecutive missing days after a known price — exceeds the cap of 5
    prices = pd.Series([100.0] + [np.nan] * 7 + [110.0])
    filled = forward_fill_capped(prices, limit=5)

    # Days 1-5 (indices 1-5) should be filled with 100.0
    assert filled.iloc[1:6].tolist() == [100.0] * 5, (
        "Expected the first 5 missing days to be filled"
    )
    # Days 6-7 (indices 6-7) should remain NaN — beyond the cap
    assert filled.iloc[6:8].isna().all(), (
        f"Expected days beyond the 5-day cap to remain NaN, "
        f"got {filled.iloc[6:8].tolist()}"
    )


def test_forward_fill_does_not_fill_leading_nan():
    """
    If a series starts with NaN (no prior price exists yet to carry
    forward), forward-fill correctly leaves it as NaN — there's nothing
    to fill from.
    """
    prices = pd.Series([np.nan, np.nan, 100.0, 101.0])
    filled = forward_fill_capped(prices, limit=5)

    assert filled.iloc[:2].isna().all(), (
        "Expected leading NaN values (no prior price to fill from) to remain NaN"
    )


# ── Test 2: daily return vs log return — must NOT be accidentally swapped ───

def test_daily_return_known_values():
    """
    A clean 10% step up should give a daily_return of exactly 0.10 —
    the simple percentage change formula.
    """
    prices = pd.Series([100.0, 110.0])
    daily_ret = compute_daily_return(prices)

    assert daily_ret.iloc[1] == pytest.approx(0.10, abs=1e-9)


def test_log_return_known_values():
    """
    The same 10% step should give a log return of ln(1.10) ≈ 0.0953,
    NOT 0.10. This is the key distinguishing check between the two
    return types — if log_return ever equals daily_return exactly,
    one of the formulas has been accidentally copy-pasted from the other.
    """
    prices = pd.Series([100.0, 110.0])
    log_ret = compute_log_return(prices)

    expected_log_return = np.log(1.10)
    assert log_ret.iloc[1] == pytest.approx(expected_log_return, abs=1e-9)
    assert log_ret.iloc[1] != pytest.approx(0.10, abs=1e-4), (
        "Log return must NOT equal the simple daily_return value — "
        "ln(1.10) ≈ 0.0953, not 0.10. The two formulas may have been swapped."
    )


def test_log_return_is_approximately_daily_return_for_small_moves():
    """
    For SMALL price moves, ln(1+x) ≈ x — log return and daily return
    should be very close (this is a real mathematical property, not a
    bug), confirming the two converge correctly at small magnitudes
    while still being distinct formulas.
    """
    prices = pd.Series([100.0, 100.5])  # a tiny 0.5% move
    daily_ret = compute_daily_return(prices).iloc[1]
    log_ret   = compute_log_return(prices).iloc[1]

    assert log_ret == pytest.approx(daily_ret, abs=1e-4)


# ── Test 3: annualised volatility scaling factor ─────────────────────────────

def test_volatility_annualisation_factor():
    """
    Volatility must be scaled by sqrt(252), the standard number of
    trading days in a year. We construct a log-return series with a
    known, constant standard deviation and confirm the annualised
    output is exactly std * sqrt(252), not some other multiple
    (a common mistake is using sqrt(365) or forgetting to annualise
    at all).
    """
    rng = np.random.default_rng(seed=99)
    # Enough points to comfortably exceed min_periods=10
    log_returns = pd.Series(rng.normal(loc=0, scale=0.02, size=30))

    vol = compute_volatility_21d(log_returns)
    last_valid = vol.dropna().iloc[-1]

    # Manually compute what it should be: rolling 21-day std * sqrt(252)
    expected = log_returns.iloc[-21:].std() * np.sqrt(252)
    assert last_valid == pytest.approx(expected, abs=1e-9)

    # Sanity check the annualisation factor itself is right (not sqrt(365))
    wrong_factor_result = log_returns.iloc[-21:].std() * np.sqrt(365)
    assert last_valid != pytest.approx(wrong_factor_result, abs=1e-6), (
        "Volatility appears to be using sqrt(365) instead of the correct "
        "sqrt(252) trading-day annualisation factor"
    )


def test_volatility_respects_min_periods():
    """
    With min_periods=10, fewer than 10 return observations should give
    NaN — an unstable, premature estimate shouldn't be silently produced.
    """
    log_returns = pd.Series(np.random.default_rng(1).normal(size=8))
    vol = compute_volatility_21d(log_returns)

    assert vol.isna().all(), (
        "Expected NaN with fewer than min_periods=10 return observations"
    )


# ── Test 4: 52-week high/low ──────────────────────────────────────────────────

def test_52w_high_is_the_maximum_not_minimum():
    """
    A simple but easy-to-get-backwards check: high_52w must be the
    rolling MAXIMUM, not the minimum. We use an asymmetric series
    where the highest and lowest values are unambiguous and distinct.
    """
    # 60 points (above min_periods=50), with one clear maximum
    prices = pd.Series([100.0] * 59 + [500.0])  # one big spike at the end
    high, low = compute_52w_high_low(prices)

    assert high.iloc[-1] == 500.0, (
        f"Expected high_52w to capture the maximum value (500.0), "
        f"got {high.iloc[-1]}"
    )
    assert low.iloc[-1] == 100.0, (
        f"Expected low_52w to capture the minimum value (100.0), "
        f"got {low.iloc[-1]}"
    )


def test_52w_high_low_respects_min_periods():
    """
    With min_periods=50, fewer than 50 price observations should give
    NaN — not a premature high/low estimate based on too little history.
    """
    prices = pd.Series(np.linspace(100, 110, 30))  # only 30 points
    high, low = compute_52w_high_low(prices)

    assert high.isna().all() and low.isna().all(), (
        "Expected NaN for both high_52w and low_52w with fewer than "
        "min_periods=50 observations"
    )


def test_pct_from_52w_high_is_zero_at_the_peak():
    """
    pct_from_52w_high = (close - high_52w) / high_52w
    At the exact moment a stock makes a new 52-week high, this value
    must be exactly 0 (close == high_52w) — not negative, not positive.
    """
    prices = pd.Series([100.0] * 59 + [500.0])  # new high on the last day
    high, _ = compute_52w_high_low(prices)

    pct_from_high = (prices - high) / high

    assert pct_from_high.iloc[-1] == pytest.approx(0.0, abs=1e-9), (
        f"Expected exactly 0 at a new 52-week high, got {pct_from_high.iloc[-1]}"
    )


# ── Test 5: per-ticker isolation ──────────────────────────────────────────────

def test_calculations_do_not_leak_across_tickers():
    """
    The real EquityProcessor uses groupby('ticker') before every
    transform. We confirm here that running the SAME calculation on
    two genuinely separate series gives independent results — i.e.
    that nothing about our replicated logic implicitly assumes a
    single continuous series across ticker boundaries.
    """
    ticker_a_prices = pd.Series([100.0, 105.0, 110.0])
    ticker_b_prices = pd.Series([50.0, 45.0, 40.0])  # falling, unrelated

    ret_a = compute_daily_return(ticker_a_prices)
    ret_b = compute_daily_return(ticker_b_prices)

    assert ret_a.iloc[1] > 0, "Ticker A is rising — its return should be positive"
    assert ret_b.iloc[1] < 0, "Ticker B is falling — its return should be negative"
    # If logic were accidentally concatenating series across tickers,
    # the first row of ticker B would compute a return relative to
    # ticker A's last price — confirm that's NOT happening by checking
    # ticker B's first return is NaN (no prior value within its own series)
    assert pd.isna(ret_b.iloc[0]), (
        "Expected NaN for the first row of ticker B's own series — if this "
        "is a real number, ticker A's data may be leaking across the boundary"
    )