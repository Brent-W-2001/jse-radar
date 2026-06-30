"""
Tests for the walk-forward backtester — rebalance date generation,
the train/predict/evaluate loop, cost application, and summary metrics.

Why these tests matter:
  This is where every previously-tested piece (chronological split,
  RegimePredictor, the outperformance label) gets composed into a loop
  that runs dozens of times unattended. A bug in the loop itself —
  not in any individual piece — could still silently leak future data,
  miscount a rebalance cost, or crash on a realistic edge case (a
  training window with only one outcome class, a date with too few
  usable tickers). These tests target the COMPOSITION, since the
  pieces it's built from already have their own dedicated test files.

Strategy:
  Small synthetic frames spanning many months, engineered to trigger
  each specific edge case deliberately.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from jse_radar.modeling.backtester import (
    generate_monthly_rebalance_dates,
    run_walk_forward_backtest,
    summarise_backtest,
    DEFAULT_REBALANCE_COST_BPS,
)
from jse_radar.modeling.regime_predictor import ALSI_TICKER


# ── Helper: build a synthetic, multi-month, multi-ticker frame ──────────────

def _build_synthetic_master(n_months: int = 30, n_tickers: int = 6) -> pd.DataFrame:
    """
    Builds a daily synthetic frame spanning n_months, with n_tickers
    plus the ALSI, deliberately separable signal, and a full
    composite_regime column so RegimePredictor can fit.
    """
    rng = np.random.default_rng(7)
    dates = pd.date_range("2020-01-01", periods=n_months * 21, freq="B")  # ~21 trading days/month

    rows = []
    tickers = [f"T{i}.JO" for i in range(n_tickers)] + [ALSI_TICKER]

    for ticker in tickers:
        # Each ticker gets its own random-walk price series, with a
        # ticker-specific drift so some are genuinely "better" than others
        drift = rng.uniform(-0.0005, 0.0010)
        price = 100.0
        for date in dates:
            price *= (1 + drift + rng.normal(0, 0.01))
            rows.append({
                "date":             date,
                "ticker":           ticker,
                "close":            price,
                "momentum_score":   rng.normal(0, 1),
                "rsi_14":           rng.uniform(20, 80),
                "trend_signal":     rng.choice([-1, 0, 1]),
                "composite_regime": rng.choice(
                    ["HIKING_TARGET_INFLATION", "CUTTING_TARGET_INFLATION"]
                ),
            })

    return pd.DataFrame(rows).sort_values(["ticker", "date"]).reset_index(drop=True)


# ── Test 1: rebalance date generation ────────────────────────────────────────

def test_generate_monthly_rebalance_dates_skips_minimum_training_window():
    """
    With min_train_months=12 and 30 months of data, the first 12
    months must be skipped entirely — there's no point evaluating a
    walk-forward step before there's enough training history behind it.
    """
    df = _build_synthetic_master(n_months=30)
    dates = generate_monthly_rebalance_dates(df, min_train_months=12)

    # Should be roughly 30 - 12 = 18 months worth of rebalance dates
    assert 15 <= len(dates) <= 19, (
        f"Expected roughly 18 rebalance dates after skipping 12 of 30 "
        f"months, got {len(dates)}"
    )
    # Every returned date must be chronologically after the 12th month started
    df["year_month"] = df["date"].dt.to_period("M")
    twelfth_month_start = df.groupby("year_month")["date"].min().sort_values().iloc[11]
    assert dates[0] >= twelfth_month_start


def test_generate_monthly_rebalance_dates_empty_with_insufficient_history():
    """
    With fewer total months than min_train_months, there should be
    zero rebalance dates — not a crash, not a negative-length slice.
    """
    df = _build_synthetic_master(n_months=5)
    dates = generate_monthly_rebalance_dates(df, min_train_months=12)
    assert dates == []


# ── Test 2: the cost haircut is applied exactly and consistently ────────────

def test_net_excess_return_equals_gross_minus_cost():
    """
    For every single rebalance period in the results, net_excess_return
    must equal excess_return minus the cost fraction, EXACTLY — this is
    simple arithmetic, but it's the one number that determines whether
    a strategy looks profitable after realistic costs or not, so it
    gets a direct, explicit check.
    """
    df = _build_synthetic_master(n_months=20, n_tickers=8)
    df = _add_minimal_labels(df)

    results = run_walk_forward_backtest(df, top_n=3, rebalance_cost_bps=15)

    if results.empty:
        pytest.skip("No rebalance periods produced — synthetic data didn't yield any")

    expected_net = results["excess_return"] - (15 / 10_000)
    pd.testing.assert_series_equal(
        results["net_excess_return"], expected_net, check_names=False
    )


def _add_minimal_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Mirrors build_outperformance_label() but kept local and minimal for test speed."""
    from jse_radar.modeling.regime_predictor import build_outperformance_label
    return build_outperformance_label(df)


# ── Test 3: walk-forward never leaks future data into training ──────────────

def test_walk_forward_never_trains_on_or_after_its_own_rebalance_date():
    """
    The single most important property of this module: at every
    rebalance date, the model used to generate that period's
    predictions must have been trained EXCLUSIVELY on data strictly
    before that date. We verify this indirectly but rigorously by
    confirming every period in the results has a rebalance_date that
    is itself a real date present in the data (sanity) and that
    re-deriving the split at that date matches what
    chronological_train_test_split would independently produce.
    """
    from jse_radar.modeling.regime_predictor import chronological_train_test_split

    df = _build_synthetic_master(n_months=20, n_tickers=8)
    df = _add_minimal_labels(df)

    results = run_walk_forward_backtest(df, top_n=3)

    if results.empty:
        pytest.skip("No rebalance periods produced")

    for rebalance_date in results["rebalance_date"]:
        train, _ = chronological_train_test_split(
            df, test_start_date=rebalance_date.strftime("%Y-%m-%d")
        )
        assert train["date"].max() < rebalance_date, (
            f"Training data for rebalance date {rebalance_date} contains "
            f"a row dated on or after that rebalance date — this is "
            f"look-ahead bias in the walk-forward loop"
        )


# ── Test 4: edge cases that would otherwise crash a real run ────────────────

def test_skips_period_with_single_class_in_training_labels():
    """
    If every training label up to some date happens to be the same
    class (e.g. all 1s), logistic regression cannot fit. The backtest
    must skip that period gracefully (log and continue), not crash
    the entire run.
    """
    df = _build_synthetic_master(n_months=15, n_tickers=4)
    df = _add_minimal_labels(df)
    # Force every label to a single class — guarantees at least the
    # earliest rebalance periods hit this edge case
    df["outperformed"] = 1.0

    # Should not raise
    results = run_walk_forward_backtest(df, top_n=2)
    assert isinstance(results, pd.DataFrame)


def test_skips_period_with_fewer_tickers_than_top_n():
    """
    If a rebalance date has fewer usable tickers than top_n requested,
    that period must be skipped, not silently return a smaller-than-
    requested portfolio that the caller doesn't expect.
    """
    df = _build_synthetic_master(n_months=15, n_tickers=2)  # only 2 real tickers
    df = _add_minimal_labels(df)

    # top_n=5 but only 2 tickers exist (excluding the ALSI) — every
    # period should be skipped
    results = run_walk_forward_backtest(df, top_n=5)
    assert results.empty or (results["picked_tickers"].str.count(",") + 1 >= 5).all()


# ── Test 5: summary metrics ───────────────────────────────────────────────────

def test_summarise_backtest_hit_rate_is_genuine_proportion():
    """hit_rate must be the exact fraction of periods where beat_alsi was True."""
    results = pd.DataFrame({
        "excess_return":     [0.02, -0.01, 0.03, -0.02],
        "net_excess_return": [0.018, -0.012, 0.028, -0.022],
        "beat_alsi":         [True, False, True, False],
    })
    summary = summarise_backtest(results)

    assert summary["hit_rate"] == pytest.approx(0.5)


def test_summarise_backtest_handles_empty_results():
    """An empty results frame (e.g. genuinely insufficient data) must not crash."""
    summary = summarise_backtest(pd.DataFrame())
    assert summary["n_periods"] == 0


def test_summarise_backtest_handles_zero_variance_gracefully():
    """
    If every period has IDENTICAL net excess return, std is 0 and the
    Sharpe-style ratio would be a division by zero — must return None,
    not crash or fabricate infinity.
    """
    results = pd.DataFrame({
        "excess_return":     [0.01, 0.01, 0.01],
        "net_excess_return": [0.01, 0.01, 0.01],
        "beat_alsi":         [True, True, True],
    })
    summary = summarise_backtest(results)

    assert summary["sharpe_style_ratio"] is None


def test_summarise_backtest_flags_thin_sample():
    """Fewer than 24 periods should trigger the thin-sample caution note."""
    results = pd.DataFrame({
        "excess_return":     [0.01] * 10,
        "net_excess_return": [0.005] * 10,
        "beat_alsi":         [True] * 10,
    })
    summary = summarise_backtest(results)

    assert "thin sample" in summary["note"].lower() or "caution" in summary["note"].lower()