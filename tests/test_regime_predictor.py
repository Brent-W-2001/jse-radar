"""
Tests for RegimePredictor — outperformance labeling, chronological
train/test splitting, and the logistic regression pipeline.

Why these tests matter:
  This is the first model in the project that touches actual stock
  returns directly, and it introduces a new class of risk beyond
  anything tested so far: a train/test split that isn't strictly
  chronological would let the model train on future data relative to
  its own test period — the same fundamental error as look-ahead bias
  in the master builder, just appearing in a different part of the
  pipeline. test_split_is_strictly_chronological is the direct
  equivalent of test_master_builder.py::test_no_look_ahead_bias and
  should be treated with the same level of scrutiny.

Strategy:
  Small, hand-built synthetic frames with known correct outputs,
  following the same approach as every other test file in this suite.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from jse_radar.modeling.regime_predictor import (
    RegimePredictor,
    build_outperformance_label,
    chronological_train_test_split,
    ALSI_TICKER,
)


# ── Test 1: outperformance label correctness ─────────────────────────────────

def test_outperformance_label_when_stock_beats_index():
    """
    A stock with a higher forward return than the ALSI on the same
    date must be labelled 1 (outperformed).
    """
    dates = pd.date_range("2024-01-01", periods=25, freq="D")
    df = pd.DataFrame({
        "date":   list(dates) * 2,
        "ticker": ["STOCK.JO"] * 25 + [ALSI_TICKER] * 25,
        # STOCK.JO rises faster than the index over the window
        "close":  list(np.linspace(100, 130, 25)) + list(np.linspace(100, 110, 25)),
    })

    labeled = build_outperformance_label(df, forward_window=10)
    stock_rows = labeled[labeled["ticker"] == "STOCK.JO"].dropna(subset=["outperformed"])

    assert (stock_rows["outperformed"] == 1).all(), (
        "A stock that genuinely rises faster than the ALSI should be "
        "labelled 1 (outperformed) on every valid row"
    )


def test_outperformance_label_when_stock_underperforms_index():
    """Mirror case: a stock rising slower than the index must be labelled 0."""
    dates = pd.date_range("2024-01-01", periods=25, freq="D")
    df = pd.DataFrame({
        "date":   list(dates) * 2,
        "ticker": ["STOCK.JO"] * 25 + [ALSI_TICKER] * 25,
        "close":  list(np.linspace(100, 105, 25)) + list(np.linspace(100, 130, 25)),
    })

    labeled = build_outperformance_label(df, forward_window=10)
    stock_rows = labeled[labeled["ticker"] == "STOCK.JO"].dropna(subset=["outperformed"])

    assert (stock_rows["outperformed"] == 0).all()


def test_outperformance_label_is_nan_near_the_end_of_data():
    """
    The last `forward_window` rows of any ticker have no future price
    to compute a forward return from — these must be NaN, not a
    fabricated 0 or 1.
    """
    dates = pd.date_range("2024-01-01", periods=15, freq="D")  # shorter than window
    df = pd.DataFrame({
        "date":   list(dates) * 2,
        "ticker": ["STOCK.JO"] * 15 + [ALSI_TICKER] * 15,
        "close":  list(np.linspace(100, 110, 15)) + list(np.linspace(100, 105, 15)),
    })

    labeled = build_outperformance_label(df, forward_window=21)  # window > data length

    assert labeled["outperformed"].isna().all(), (
        "With insufficient future data for the entire series, every "
        "label should be NaN"
    )


# ── Test 2: chronological split — the most important test in this file ──────

def test_split_is_strictly_chronological():
    """
    THE critical correctness property: no row in train may have a date
    on or after any row in test, and no row in test may have a date
    before test_start_date. This is the direct equivalent of the
    no-look-ahead-bias test for the master builder's asof merge.
    """
    dates = pd.date_range("2020-01-01", periods=200, freq="D")
    df = pd.DataFrame({
        "date":   dates,
        "ticker": ["STOCK.JO"] * 200,
        "value":  np.arange(200),
    })

    cutoff = "2020-04-01"
    train, test = chronological_train_test_split(df, test_start_date=cutoff)

    assert len(train) + len(test) == len(df), (
        "Split must partition every row — none lost, none duplicated"
    )
    assert train["date"].max() < pd.Timestamp(cutoff), (
        f"Train set contains a date on or after the cutoff ({cutoff}) — "
        f"this would leak future information into training"
    )
    assert test["date"].min() >= pd.Timestamp(cutoff), (
        f"Test set contains a date before the cutoff ({cutoff})"
    )
    # The single most direct possible statement of the property:
    assert train["date"].max() < test["date"].min(), (
        "Every train date must be strictly earlier than every test date"
    )


def test_split_with_cutoff_outside_data_range():
    """
    If the cutoff is before all data, train should be empty and test
    should contain everything — and vice versa. Must not crash on
    these boundary cases.
    """
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    df = pd.DataFrame({"date": dates, "value": range(10)})

    train, test = chronological_train_test_split(df, test_start_date="2023-01-01")
    assert len(train) == 0 and len(test) == 10

    train, test = chronological_train_test_split(df, test_start_date="2025-01-01")
    assert len(train) == 10 and len(test) == 0


# ── Test 3: regime one-hot encoding stays consistent between train/test ─────

def _make_synthetic_frame(regimes: list[str], n_per_regime: int = 50) -> pd.DataFrame:
    """Builds a synthetic, fittable frame with a deliberately separable signal."""
    rng = np.random.default_rng(42)
    rows = []
    for regime in regimes:
        for _ in range(n_per_regime):
            momentum = rng.normal(0, 1)
            rows.append({
                "momentum_score":   momentum,
                "rsi_14":           rng.uniform(20, 80),
                "trend_signal":     rng.choice([-1, 0, 1]),
                "composite_regime": regime,
                # Deliberately separable: positive momentum -> outperformed=1
                "outperformed":     1 if momentum > 0 else 0,
            })
    return pd.DataFrame(rows)


def test_predict_works_when_test_set_has_unseen_regime():
    """
    If the test set contains a regime that never appeared during
    training (a real possibility — e.g. the macro environment enters
    a genuinely new state after the model was last fit), predict_proba
    must still work, treating that regime's one-hot columns as all
    zero rather than crashing or misaligning columns.
    """
    train_df = _make_synthetic_frame(["HIKING_TARGET_INFLATION", "CUTTING_TARGET_INFLATION"])
    model = RegimePredictor().fit(train_df)

    test_df = _make_synthetic_frame(["HIKING_LOW_INFLATION"], n_per_regime=10)
    # Should not raise, despite HIKING_LOW_INFLATION never being seen at fit time
    probs = model.predict_proba(test_df)

    assert len(probs) == len(test_df)
    assert probs.notna().all(), "Unseen regime should still produce a valid probability"


def test_predict_works_when_training_regime_missing_from_test():
    """
    Mirror case: if the training data had 3 regimes but the test set
    only contains 1 of them, the feature matrix must still align
    correctly — the missing regime columns become structurally zero
    for every test row, not cause a column-count mismatch.
    """
    train_df = _make_synthetic_frame(
        ["HIKING_TARGET_INFLATION", "CUTTING_TARGET_INFLATION", "HIKING_HIGH_INFLATION"]
    )
    model = RegimePredictor().fit(train_df)

    test_df = _make_synthetic_frame(["HIKING_TARGET_INFLATION"], n_per_regime=10)
    probs = model.predict_proba(test_df)

    assert len(probs) == len(test_df)
    assert probs.notna().all()


# ── Test 4: NaN handling ──────────────────────────────────────────────────────

def test_fit_drops_rows_with_missing_features_not_crash():
    """
    Rows with NaN in any feature or the label must be excluded from
    fitting, not crash sklearn and not be silently zero-filled (which
    would fabricate information the model never actually had).
    """
    df = _make_synthetic_frame(["HIKING_TARGET_INFLATION"], n_per_regime=30)
    df.loc[0:5, "momentum_score"] = np.nan  # corrupt some rows

    # Should not raise
    model = RegimePredictor().fit(df)
    assert model.model is not None


def test_predict_proba_returns_nan_for_rows_with_missing_features():
    """At prediction time, a row missing a required feature must get NaN, not a fabricated probability."""
    train_df = _make_synthetic_frame(["HIKING_TARGET_INFLATION"], n_per_regime=50)
    model = RegimePredictor().fit(train_df)

    test_df = _make_synthetic_frame(["HIKING_TARGET_INFLATION"], n_per_regime=5)
    test_df.loc[0, "rsi_14"] = np.nan

    probs = model.predict_proba(test_df)
    assert pd.isna(probs.iloc[0]), "Row with missing feature should predict NaN"
    assert probs.iloc[1:].notna().all(), "Rows with complete features should still predict normally"


# ── Test 5: the model learns SOMETHING on an obviously separable dataset ────

def test_model_learns_the_obviously_separable_signal():
    """
    Not a test of real predictive power — a sanity check that the
    pipeline isn't structurally broken. On data where positive
    momentum_score deterministically means outperformed=1 (by
    construction in _make_synthetic_frame), the fitted model must
    assign a materially higher average probability to genuinely
    positive-momentum rows than to negative-momentum rows.
    """
    df = _make_synthetic_frame(["HIKING_TARGET_INFLATION"], n_per_regime=300)
    model = RegimePredictor().fit(df)

    probs = model.predict_proba(df)

    positive_momentum_probs = probs[df["momentum_score"] > 0]
    negative_momentum_probs = probs[df["momentum_score"] < 0]

    assert positive_momentum_probs.mean() > negative_momentum_probs.mean() + 0.2, (
        f"Expected materially higher predicted probability for the "
        f"positive-momentum group on an obviously separable synthetic "
        f"dataset. Got {positive_momentum_probs.mean():.3f} vs "
        f"{negative_momentum_probs.mean():.3f} — the pipeline may be broken"
    )


def test_coefficient_summary_returns_one_row_per_feature():
    """
    coefficient_summary() should return exactly one row per feature
    (signal columns + regime dummy columns), with no duplicates and
    no missing features.
    """
    df = _make_synthetic_frame(["HIKING_TARGET_INFLATION", "CUTTING_TARGET_INFLATION"])
    model = RegimePredictor().fit(df)

    summary = model.coefficient_summary()
    expected_n = len(model.feature_columns) + len(model.regime_columns)

    assert len(summary) == expected_n
    assert summary["feature"].nunique() == expected_n  # no duplicates


# ── Test 6: calling predict before fit raises clearly ───────────────────────

def test_predict_before_fit_raises_runtime_error():
    model = RegimePredictor()
    df = _make_synthetic_frame(["HIKING_TARGET_INFLATION"], n_per_regime=5)

    with pytest.raises(RuntimeError, match="fit"):
        model.predict_proba(df)


def test_coefficient_summary_before_fit_raises_runtime_error():
    model = RegimePredictor()
    with pytest.raises(RuntimeError, match="fit"):
        model.coefficient_summary()