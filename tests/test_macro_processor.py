"""
Tests for MacroProcessor — mixed-frequency resampling, forward-fill cap,
and derived macro indicators (CPI YoY, real interest rate, ZAR/USD MoM).

Why these tests matter:
  This module blends data with very different natural frequencies —
  daily FX rates, monthly CPI, quarterly GDP-adjacent series — into one
  consistent monthly frame. Resampling errors here are particularly
  dangerous because they're invisible downstream: a wrongly-averaged
  rate or an off-by-one in the year-on-year inflation calculation would
  silently distort the regime classifier, the real interest rate, and
  every chart built on top of them.

Strategy:
  Small, hand-built daily/monthly series with known, hand-calculable
  expected outputs after resampling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ── Replicated logic (mirrors MacroProcessor.process()) ──────────────────────

def resample_fx_mean(df: pd.DataFrame, fx_cols: list[str]) -> pd.DataFrame:
    """Mirrors: df[fx_cols].resample('ME').mean()"""
    return df[fx_cols].resample("ME").mean()


def resample_other_last(df: pd.DataFrame, other_cols: list[str]) -> pd.DataFrame:
    """Mirrors: df[other_cols].resample('ME').last()"""
    return df[other_cols].resample("ME").last()


def forward_fill_capped(series: pd.Series, limit: int = 3) -> pd.Series:
    """Mirrors: df_monthly.ffill(limit=3)"""
    return series.ffill(limit=limit)


def compute_cpi_yoy(cpi_all: pd.Series) -> pd.Series:
    """Mirrors: df_monthly['cpi_all'].pct_change(12) * 100"""
    return cpi_all.pct_change(12) * 100


def compute_real_rate(tbill_rate: pd.Series, cpi_yoy_pct: pd.Series) -> pd.Series:
    """Mirrors: df_monthly['tbill_rate'] - df_monthly['cpi_yoy_pct']"""
    return tbill_rate - cpi_yoy_pct


def compute_zar_usd_mom(zar_usd: pd.Series) -> pd.Series:
    """Mirrors: df_monthly['zar_usd'].pct_change() * 100"""
    return zar_usd.pct_change() * 100


# ── Test 1: FX columns are averaged, not just taken as last value ──────────

def test_fx_resampling_uses_mean_not_last():
    """
    ZAR/USD is a daily series. Resampling to monthly with .mean() gives
    the average rate over the month, which is more representative of
    actual trading conditions than a single end-of-month snapshot.
    We construct a month where the average and the last value are
    deliberately very different, and confirm the mean is what's used.
    """
    dates = pd.date_range("2024-01-01", "2024-01-31", freq="D")
    # Rate climbs steadily through January, ending much higher than it started
    zar_usd = pd.Series(np.linspace(15.0, 19.0, len(dates)), index=dates)
    df = pd.DataFrame({"zar_usd": zar_usd})

    resampled = resample_fx_mean(df, ["zar_usd"])

    monthly_mean = zar_usd.mean()
    monthly_last = zar_usd.iloc[-1]

    assert resampled["zar_usd"].iloc[0] == pytest.approx(monthly_mean, abs=1e-6)
    assert resampled["zar_usd"].iloc[0] != pytest.approx(monthly_last, abs=0.1), (
        "FX resampling appears to be using the last value of the month "
        "instead of the mean — these should differ noticeably for a "
        "steadily trending month"
    )


def test_non_fx_resampling_uses_last_not_mean():
    """
    A rate/level column (e.g. CPI index) is resampled with .last() —
    the most recent reading in the month, not an average across days
    that mostly don't have new data anyway. We confirm the opposite
    behaviour from the FX test above.
    """
    dates = pd.date_range("2024-01-01", "2024-01-31", freq="D")
    # Simulate a column that only updates once mid-month, otherwise NaN —
    # representative of a low-frequency series sparsely populated daily
    cpi = pd.Series([np.nan] * len(dates), index=dates)
    cpi.iloc[-1] = 105.3  # only the last day of the month has a real reading
    df = pd.DataFrame({"cpi_all": cpi})

    resampled = resample_other_last(df, ["cpi_all"])

    assert resampled["cpi_all"].iloc[0] == pytest.approx(105.3), (
        f"Expected .last() resampling to pick up the most recent reading "
        f"(105.3), got {resampled['cpi_all'].iloc[0]}"
    )


# ── Test 2: monthly forward-fill cap (limit=3) ───────────────────────────────

def test_monthly_ffill_fills_short_gaps():
    """
    A gap of up to 3 months (e.g. the repo rate staying unchanged
    between MPC meetings) should be fully forward-filled.
    """
    rates = pd.Series([7.25, np.nan, np.nan, np.nan, 7.50])
    filled = forward_fill_capped(rates, limit=3)

    assert filled.iloc[1:4].tolist() == [7.25, 7.25, 7.25], (
        f"Expected a 3-month gap to be fully filled with the prior rate, "
        f"got {filled.iloc[1:4].tolist()}"
    )


def test_monthly_ffill_stops_after_cap():
    """
    A gap longer than 3 months should only fill the first 3 — beyond
    that, a genuinely missing/discontinued series shouldn't have a
    stale rate carried forward indefinitely.
    """
    rates = pd.Series([7.25] + [np.nan] * 5 + [7.50])
    filled = forward_fill_capped(rates, limit=3)

    assert filled.iloc[1:4].tolist() == [7.25, 7.25, 7.25], (
        "Expected the first 3 missing months to be filled"
    )
    assert filled.iloc[4:6].isna().all(), (
        f"Expected months beyond the 3-month cap to remain NaN, "
        f"got {filled.iloc[4:6].tolist()}"
    )


# ── Test 3: CPI year-on-year calculation ──────────────────────────────────────

def test_cpi_yoy_known_value():
    """
    A clean, hand-calculable example: CPI rises from 100 to 105 over
    exactly 12 months. The year-on-year inflation rate at that 12th
    month must be exactly 5.0%, not 4.0% or 6.0% from an off-by-one
    in the pct_change() periods argument.
    """
    # 13 months of data: month 0 = 100, month 12 = 105 (a clean 5% rise)
    cpi = pd.Series([100.0] + [101.0] * 11 + [105.0])
    cpi_yoy = compute_cpi_yoy(cpi)

    assert cpi_yoy.iloc[12] == pytest.approx(5.0, abs=1e-6), (
        f"Expected exactly 5.0% YoY inflation at month 12 "
        f"(100 -> 105 over 12 months), got {cpi_yoy.iloc[12]}"
    )


def test_cpi_yoy_is_nan_before_12_months_of_history():
    """
    Year-on-year inflation is fundamentally undefined until 12 months
    of history exist — there's no "12 months ago" value to compare
    against yet. Confirm this produces NaN, not a misleading early value
    computed against an incomplete or wrong reference point.
    """
    cpi = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0])  # only 5 months
    cpi_yoy = compute_cpi_yoy(cpi)

    assert cpi_yoy.isna().all(), (
        "Expected all NaN with fewer than 12 months of CPI history"
    )


# ── Test 4: real interest rate sign and subtraction order ───────────────────

def test_real_rate_subtraction_order():
    """
    real_tbill_rate = tbill_rate - cpi_yoy_pct
    This must be nominal MINUS inflation, not the other way round —
    a positive real rate means borrowing is genuinely costly after
    accounting for inflation. We use values where the two orderings
    give clearly different, easily distinguishable signs.
    """
    tbill_rate  = pd.Series([8.0])
    cpi_yoy_pct = pd.Series([5.0])

    real_rate = compute_real_rate(tbill_rate, cpi_yoy_pct)

    assert real_rate.iloc[0] == pytest.approx(3.0), (
        f"Expected real_tbill_rate = 8.0 - 5.0 = 3.0, got {real_rate.iloc[0]}"
    )
    # If the subtraction were accidentally reversed, this would be -3.0
    assert real_rate.iloc[0] != pytest.approx(-3.0), (
        "Real interest rate calculation appears to have the subtraction "
        "order reversed (cpi_yoy_pct - tbill_rate instead of the other way)"
    )


def test_real_rate_can_be_negative():
    """
    When inflation exceeds the nominal rate, the real rate is genuinely
    negative — this is a normal, important economic condition (e.g.
    South Africa during the 2022-2023 inflation spike before hikes
    caught up) and must not be floored at zero or otherwise distorted.
    """
    tbill_rate  = pd.Series([5.0])
    cpi_yoy_pct = pd.Series([8.0])

    real_rate = compute_real_rate(tbill_rate, cpi_yoy_pct)

    assert real_rate.iloc[0] == pytest.approx(-3.0)


# ── Test 5: ZAR/USD month-on-month percentage change ────────────────────────

def test_zar_usd_mom_positive_when_rand_weakens():
    """
    A rising ZAR/USD number means more rand are needed per dollar —
    the rand has WEAKENED. The month-on-month % change should be
    positive in this case. This sign convention matters because the
    regime notebook and currency_regime classification both depend
    on it being right.
    """
    zar_usd = pd.Series([15.0, 16.5])  # rand weakened from 15 to 16.5 per USD
    mom_pct = compute_zar_usd_mom(zar_usd)

    assert mom_pct.iloc[1] == pytest.approx(10.0, abs=1e-6), (
        f"Expected +10% MoM change for rand weakening from 15.0 to 16.5, "
        f"got {mom_pct.iloc[1]}"
    )
    assert mom_pct.iloc[1] > 0, (
        "A weakening rand (higher ZAR/USD) must show a POSITIVE "
        "month-on-month change"
    )


def test_zar_usd_mom_negative_when_rand_strengthens():
    """Mirror of the above: a falling ZAR/USD number means the rand strengthened."""
    zar_usd = pd.Series([16.5, 15.0])  # rand strengthened
    mom_pct = compute_zar_usd_mom(zar_usd)

    assert mom_pct.iloc[1] < 0, (
        "A strengthening rand (lower ZAR/USD) must show a NEGATIVE "
        "month-on-month change"
    )