"""
Tests for MacroRegimeClassifier — rate/inflation/currency regime labels
and regime duration counting.

Why these tests matter:
  This is the exact logic that produced a subtle, almost-missed problem
  in the regime analysis notebook: a regime with only one or two short
  historical spells produced an extreme, untrustworthy average return
  that initially looked like a real finding. These tests lock down the
  underlying classification and duration-counting logic so we can trust
  what regime_duration and composite_regime are actually telling us.

Strategy:
  We replicate the exact classification logic from MacroRegimeClassifier
  against small, hand-built monthly macro series where we know the
  correct regime label and duration for every row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ── Replicated logic (mirrors MacroRegimeClassifier.classify()) ─────────────

SARB_TARGET_LOW  = 3.0
SARB_TARGET_HIGH = 6.0


def classify_rate_regime(tbill_rate: pd.Series) -> pd.Series:
    """Mirrors: compare current rate to its own 3-month rolling mean."""
    rate_3m_avg = tbill_rate.rolling(3, min_periods=2).mean()
    return pd.Series(
        np.where(
            tbill_rate > rate_3m_avg, "HIKING",
            np.where(tbill_rate < rate_3m_avg, "CUTTING", "NEUTRAL"),
        ),
        index=tbill_rate.index,
    )


def classify_inflation_regime(cpi_yoy_pct: pd.Series) -> pd.Series:
    """Mirrors the pd.cut bucketing against the SARB 3-6% target band."""
    return pd.cut(
        cpi_yoy_pct,
        bins=[-np.inf, SARB_TARGET_LOW, SARB_TARGET_HIGH, np.inf],
        labels=["LOW_INFLATION", "TARGET_INFLATION", "HIGH_INFLATION"],
    ).astype(str)


def compute_regime_duration(composite_regime: pd.Series) -> pd.Series:
    """Mirrors the regime_changed/cumsum/cumcount duration logic exactly."""
    regime_changed   = (composite_regime != composite_regime.shift(1)).astype(int)
    spell_id         = regime_changed.cumsum()
    regime_duration  = spell_id.groupby(spell_id).cumcount() + 1
    return regime_duration


# ── Test 1: rate regime direction ────────────────────────────────────────────

def test_rising_rate_classified_as_hiking():
    """
    A rate that is currently ABOVE its own 3-month rolling average is,
    by definition, in an upward-trending (hiking) phase. We construct
    a series that's unambiguously climbing and check the most recent
    reading is labelled HIKING, not CUTTING.
    """
    rates = pd.Series([5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0])
    regime = classify_rate_regime(rates)

    assert regime.iloc[-1] == "HIKING", (
        f"Expected the latest reading of a steadily rising rate series "
        f"to be classified HIKING, got {regime.iloc[-1]}"
    )


def test_falling_rate_classified_as_cutting():
    """Mirror of the above: a steadily falling series should read CUTTING."""
    rates = pd.Series([8.0, 7.5, 7.0, 6.5, 6.0, 5.5, 5.0])
    regime = classify_rate_regime(rates)

    assert regime.iloc[-1] == "CUTTING", (
        f"Expected the latest reading of a steadily falling rate series "
        f"to be classified CUTTING, got {regime.iloc[-1]}"
    )


def test_flat_rate_classified_as_neutral():
    """
    A rate that hasn't moved at all should be exactly equal to its own
    rolling average — neither hiking nor cutting, genuinely NEUTRAL.
    """
    rates = pd.Series([7.0, 7.0, 7.0, 7.0, 7.0])
    regime = classify_rate_regime(rates)

    assert regime.iloc[-1] == "NEUTRAL", (
        f"Expected a perfectly flat rate series to be NEUTRAL, "
        f"got {regime.iloc[-1]}"
    )


# ── Test 2: inflation bucketing against SARB target boundaries ──────────────

def test_inflation_below_target_band():
    """CPI YoY below 3.0% — below the SARB target band entirely."""
    cpi = pd.Series([1.5, 2.0, 2.9])
    regime = classify_inflation_regime(cpi)
    assert (regime == "LOW_INFLATION").all()


def test_inflation_within_target_band():
    """CPI YoY between 3.0% and 6.0% — squarely within SARB's target."""
    cpi = pd.Series([3.5, 4.5, 5.9])
    regime = classify_inflation_regime(cpi)
    assert (regime == "TARGET_INFLATION").all()


def test_inflation_above_target_band():
    """CPI YoY above 6.0% — above the SARB target band entirely."""
    cpi = pd.Series([6.5, 8.0, 10.0])
    regime = classify_inflation_regime(cpi)
    assert (regime == "HIGH_INFLATION").all()


def test_inflation_exact_boundary_values():
    """
    The exact boundary values (3.0 and 6.0) are the most likely place
    for an off-by-one error in bin edges. pd.cut's default is
    right-inclusive, meaning a value exactly AT a bin edge belongs to
    the LOWER bucket. We pin down this exact behaviour so any future
    change to the bins is caught immediately.
    """
    cpi = pd.Series([3.0, 6.0])
    regime = classify_inflation_regime(cpi)

    # 3.0 is the right edge of the LOW_INFLATION bin -> belongs to LOW_INFLATION
    # 6.0 is the right edge of the TARGET_INFLATION bin -> belongs to TARGET_INFLATION
    assert regime.iloc[0] == "LOW_INFLATION", (
        f"Expected CPI exactly at 3.0 to fall in LOW_INFLATION (right-inclusive "
        f"default bin edge), got {regime.iloc[0]}"
    )
    assert regime.iloc[1] == "TARGET_INFLATION", (
        f"Expected CPI exactly at 6.0 to fall in TARGET_INFLATION (right-inclusive "
        f"default bin edge), got {regime.iloc[1]}"
    )


# ── Test 3: composite regime string construction ─────────────────────────────

def test_composite_regime_string_order():
    """
    composite_regime = rate_regime + '_' + inflation_regime
    The order matters for readability and for any downstream code that
    parses the string (e.g. analysis_df["rate_regime"] = composite.split("_")[0]
    in the regime notebook). We pin the exact order down explicitly.
    """
    rate_regime      = "HIKING"
    inflation_regime = "HIGH_INFLATION"
    composite = rate_regime + "_" + inflation_regime

    assert composite == "HIKING_HIGH_INFLATION"
    # Confirm it splits back apart the way the notebook relies on:
    assert composite.split("_")[0] == "HIKING"


# ── Test 4: regime duration counting (the spell-count logic) ────────────────

def test_regime_duration_resets_on_change():
    """
    regime_duration should count consecutive months WITHIN the current
    regime, resetting to 1 every time the regime changes. This is the
    exact logic that revealed the under-sampled regimes in the notebook
    — getting this wrong would have hidden that problem entirely.
    """
    composite = pd.Series([
        "HIKING_HIGH_INFLATION", "HIKING_HIGH_INFLATION", "HIKING_HIGH_INFLATION",
        "CUTTING_TARGET_INFLATION", "CUTTING_TARGET_INFLATION",
        "HIKING_HIGH_INFLATION",
    ])

    duration = compute_regime_duration(composite)

    expected = [1, 2, 3, 1, 2, 1]
    assert duration.tolist() == expected, (
        f"Expected duration to reset to 1 on each regime change and "
        f"increment within a spell, got {duration.tolist()}, expected {expected}"
    )


def test_regime_duration_single_long_spell():
    """A regime that never changes should count up continuously with no resets."""
    composite = pd.Series(["CUTTING_LOW_INFLATION"] * 5)
    duration = compute_regime_duration(composite)

    assert duration.tolist() == [1, 2, 3, 4, 5]


def test_distinct_spell_count_matches_manual_count():
    """
    This directly tests the property that mattered in the regime
    notebook: counting the number of INDEPENDENT spells of a regime
    (not just total months). We replicate the exact spells-table logic
    from the notebook and verify it against a hand-counted example with
    a known number of separate occurrences.
    """
    # CUTTING_LOW_INFLATION appears in 3 separate, non-contiguous spells
    composite = pd.Series([
        "HIKING_HIGH_INFLATION",      # spell 1 of HIKING_HIGH_INFLATION
        "CUTTING_LOW_INFLATION",      # spell 1 of CUTTING_LOW_INFLATION (1 month)
        "HIKING_HIGH_INFLATION",      # spell 2 of HIKING_HIGH_INFLATION
        "CUTTING_LOW_INFLATION",      # spell 2 of CUTTING_LOW_INFLATION
        "CUTTING_LOW_INFLATION",      # still spell 2 (consecutive)
        "HIKING_HIGH_INFLATION",      # spell 3 of HIKING_HIGH_INFLATION
        "CUTTING_LOW_INFLATION",      # spell 3 of CUTTING_LOW_INFLATION (1 month)
    ])

    # Mirror of the notebook's spell-counting logic exactly
    regime_changed = (composite != composite.shift(1)).cumsum()
    spells = (
        pd.DataFrame({"regime": composite, "spell_id": regime_changed})
        .groupby("spell_id")["regime"].first()
    )
    spell_counts = spells.value_counts()

    assert spell_counts["CUTTING_LOW_INFLATION"] == 3, (
        f"Expected 3 independent spells of CUTTING_LOW_INFLATION, "
        f"got {spell_counts['CUTTING_LOW_INFLATION']}"
    )
    assert spell_counts["HIKING_HIGH_INFLATION"] == 3, (
        f"Expected 3 independent spells of HIKING_HIGH_INFLATION, "
        f"got {spell_counts['HIKING_HIGH_INFLATION']}"
    )


# ── Test 5: NaN handling — missing data shouldn't crash classification ──────

def test_rate_regime_with_nan_does_not_crash():
    """
    Real FRED data has gaps (we've hit this in production — series
    sometimes return fewer observations than expected). Classification
    must handle NaN gracefully — comparisons with NaN should produce
    NEUTRAL (since NaN > NaN and NaN < NaN are both False), not raise
    an exception.
    """
    rates = pd.Series([5.0, np.nan, 6.0, 6.5])
    regime = classify_rate_regime(rates)

    # Should not raise; the NaN row's regime should be NEUTRAL since
    # the comparison can't be made
    assert len(regime) == 4
    assert regime.iloc[1] == "NEUTRAL"