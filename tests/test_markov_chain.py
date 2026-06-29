"""
Tests for RegimeMarkovChain — transition matrix estimation, confidence
flagging, and expected duration calculation.

Why these tests matter:
  A transition matrix can look authoritative — clean rows of percentages
  summing to 100% — while being built on almost no real evidence for the
  rarer regimes. These tests prove the counting and normalisation logic
  is exactly correct on a known sequence, AND that the confidence-flagging
  mechanism actually distinguishes well-supported transitions from
  thin ones, continuing the same discipline as the regime notebook's
  spell-count filter.

Strategy:
  A single small, fully hand-traceable regime sequence is used across
  most tests, so the expected transition counts can be verified by eye
  rather than trusting a separate "ground truth" calculation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from jse_radar.modeling.markov_chain import (
    RegimeMarkovChain,
    MIN_OCCURRENCES_FOR_CONFIDENCE,
)


# ── Fixture: a small, fully hand-traceable regime sequence ──────────────────
#
# Sequence (12 months):
#   A, A, B, A, A, B, A, B, B, A, A, B
#
# Transitions (11 total, each row consumed in order):
#   A->A, A->B, B->A, A->A, A->B, B->A, A->B, B->B, B->A, A->A, A->B
#
# Manually counted FROM "A" (7 times as a from-state):
#   A->A: 3  (positions 0->1, 3->4, 9->10)
#   A->B: 4  (positions 1->2, 4->5, 6->7, 10->11)
#   total from A = 7  ->  P(A->A) = 3/7, P(A->B) = 4/7
#
# Manually counted FROM "B" (4 times as a from-state):
#   B->A: 3  (positions 2->3, 5->6, 8->9)
#   B->B: 1  (positions 7->8)
#   total from B = 4  ->  P(B->A) = 3/4, P(B->B) = 1/4

@pytest.fixture
def simple_sequence() -> pd.Series:
    return pd.Series(["A", "A", "B", "A", "A", "B", "A", "B", "B", "A", "A", "B"])


# ── Test 1: transition counts and probabilities are exactly correct ─────────

def test_transition_probabilities_match_hand_calculation(simple_sequence):
    chain = RegimeMarkovChain().fit(simple_sequence)

    assert chain.transition_matrix.loc["A", "A"] == pytest.approx(3 / 7)
    assert chain.transition_matrix.loc["A", "B"] == pytest.approx(4 / 7)
    assert chain.transition_matrix.loc["B", "A"] == pytest.approx(3 / 4)
    assert chain.transition_matrix.loc["B", "B"] == pytest.approx(1 / 4)


def test_occurrence_counts_match_hand_calculation(simple_sequence):
    chain = RegimeMarkovChain().fit(simple_sequence)

    assert chain.occurrence_counts["A"] == 7
    assert chain.occurrence_counts["B"] == 4


# ── Test 2: every row sums to exactly 1 ──────────────────────────────────────

def test_every_row_sums_to_one(simple_sequence):
    """
    Each row is a probability distribution over next-states — it must
    sum to exactly 1 (within floating point tolerance), never more or
    less, regardless of how many distinct regimes exist.
    """
    chain = RegimeMarkovChain().fit(simple_sequence)
    row_sums = chain.transition_matrix.sum(axis=1)

    assert row_sums.tolist() == pytest.approx([1.0, 1.0])


def test_matrix_is_square_with_all_observed_regimes(simple_sequence):
    """The matrix must have every distinct regime as both a row and a column."""
    chain = RegimeMarkovChain().fit(simple_sequence)

    assert sorted(chain.transition_matrix.index) == ["A", "B"]
    assert sorted(chain.transition_matrix.columns) == ["A", "B"]


# ── Test 3: a regime that's never a FROM-state doesn't crash or fabricate ───

def test_regime_only_at_the_end_has_no_transition_row():
    """
    A regime that appears ONLY as the final observation in the sequence
    was never followed by anything — there's genuinely no transition
    data for it. This must produce an all-NaN row, not a divide-by-zero
    crash and not a fabricated uniform distribution.
    """
    # "C" appears exactly once, as the very last observation
    sequence = pd.Series(["A", "A", "B", "A", "C"])
    chain = RegimeMarkovChain().fit(sequence)

    assert chain.occurrence_counts["C"] == 0
    assert chain.transition_matrix.loc["C"].isna().all(), (
        "A regime that was never observed as a FROM-state should have "
        "an all-NaN transition row, not a fabricated probability"
    )


def test_predicting_from_never_observed_from_state_raises_clear_error():
    """
    Calling predict_next_regime_probabilities() on a regime with an
    all-NaN row (never seen as a FROM-state) must raise a clear,
    informative error rather than returning NaN values silently.
    """
    sequence = pd.Series(["A", "A", "B", "A", "C"])
    chain = RegimeMarkovChain().fit(sequence)

    with pytest.raises(ValueError, match="never"):
        chain.predict_next_regime_probabilities("C")


def test_predicting_from_unseen_regime_raises_clear_error(simple_sequence):
    """
    Asking for predictions starting from a regime that doesn't exist
    anywhere in the training data at all must raise immediately,
    not silently look up something incorrect via fuzzy matching.
    """
    chain = RegimeMarkovChain().fit(simple_sequence)

    with pytest.raises(ValueError, match="never observed"):
        chain.predict_next_regime_probabilities("NONEXISTENT_REGIME")


# ── Test 4: predict_next_regime_probabilities returns a sorted distribution ──

def test_predict_returns_probabilities_sorted_descending(simple_sequence):
    """
    The predicted distribution should be sorted with the most likely
    next regime first — this is what makes the output directly useful
    without the caller having to sort it themselves.
    """
    chain = RegimeMarkovChain().fit(simple_sequence)
    result = chain.predict_next_regime_probabilities("A")

    assert result.index[0] == "B"  # P(A->B) = 4/7 > P(A->A) = 3/7
    assert result.iloc[0] == pytest.approx(4 / 7)
    assert list(result) == sorted(result, reverse=True)


# ── Test 5: confidence flagging ───────────────────────────────────────────────

def test_confidence_flags_low_occurrence_regime_as_unreliable():
    """
    A regime observed fewer times than MIN_OCCURRENCES_FOR_CONFIDENCE
    as a FROM-state must be flagged as unreliable — mirrors the
    MIN_SPELLS discipline from the regime analysis notebook, applied
    here to raw occurrence count.
    """
    # "RARE" appears as a from-state only twice — well below the threshold
    sequence = pd.Series(
        ["COMMON"] * (MIN_OCCURRENCES_FOR_CONFIDENCE + 5) + ["RARE", "COMMON", "RARE", "COMMON"]
    )
    chain = RegimeMarkovChain().fit(sequence)

    confidence = chain.confidence_for("RARE")

    assert confidence["reliable"] is False
    assert confidence["historical_occurrences"] == 2


def test_confidence_flags_high_occurrence_regime_as_reliable():
    """The mirror case: enough occurrences should be flagged reliable."""
    sequence = pd.Series(["COMMON", "OTHER"] * (MIN_OCCURRENCES_FOR_CONFIDENCE + 5))
    chain = RegimeMarkovChain().fit(sequence)

    confidence = chain.confidence_for("COMMON")

    assert confidence["reliable"] is True
    assert confidence["historical_occurrences"] >= MIN_OCCURRENCES_FOR_CONFIDENCE


# ── Test 6: expected regime duration ─────────────────────────────────────────

def test_expected_duration_known_value():
    """
    For a regime with self-transition probability p_stay = 0.75,
    expected duration = 1 / (1 - 0.75) = 4.0 months — a clean,
    hand-checkable geometric-distribution result.

    We construct a sequence where "STICKY" stays "STICKY" exactly 3
    times out of 4 opportunities (p_stay = 0.75).
    """
    # STICKY -> STICKY (x3), STICKY -> OTHER (x1), then pad with OTHER->STICKY
    # to make every transition well-defined
    sequence = pd.Series([
        "STICKY", "STICKY", "STICKY", "STICKY", "OTHER",
        "STICKY", "STICKY", "STICKY", "STICKY", "OTHER",
    ])
    chain = RegimeMarkovChain().fit(sequence)

    # p_stay for STICKY: out of 8 times STICKY was a from-state,
    # how many times did it transition to itself again?
    # positions: 0->1 (S), 1->2 (S), 2->3 (S), 3->4 (O),
    #            5->6 (S), 6->7 (S), 7->8 (S), 8->9 (O)
    # = 6 self-transitions out of 8 from-states -> p_stay = 0.75
    expected = 1 / (1 - 0.75)
    assert chain.expected_regime_duration("STICKY") == pytest.approx(expected)


def test_expected_duration_is_infinite_when_regime_never_leaves():
    """
    If, within the sample, a regime is ALWAYS followed by itself
    (p_stay = 1.0), the geometric-distribution formula gives infinite
    expected duration. This is mathematically correct given the data
    (the regime is "absorbing" in this sample) and must be returned
    as float('inf'), not raise a ZeroDivisionError.
    """
    sequence = pd.Series(["LOCKED"] * 10)  # never transitions to anything else
    chain = RegimeMarkovChain().fit(sequence)

    assert chain.expected_regime_duration("LOCKED") == float("inf")


def test_expected_duration_is_nan_for_never_observed_from_state():
    """
    A regime with no transition data at all (all-NaN row) should
    propagate that NaN into expected_regime_duration rather than
    erroring or fabricating a number.
    """
    sequence = pd.Series(["A", "A", "B", "A", "C"])  # C only at the end
    chain = RegimeMarkovChain().fit(sequence)

    assert pd.isna(chain.expected_regime_duration("C"))


# ── Test 7: calling methods before fit() raises clearly ─────────────────────

def test_predict_before_fit_raises_runtime_error():
    chain = RegimeMarkovChain()
    with pytest.raises(RuntimeError, match="fit"):
        chain.predict_next_regime_probabilities("A")


def test_confidence_before_fit_raises_runtime_error():
    chain = RegimeMarkovChain()
    with pytest.raises(RuntimeError, match="fit"):
        chain.confidence_for("A")


def test_expected_duration_before_fit_raises_runtime_error():
    chain = RegimeMarkovChain()
    with pytest.raises(RuntimeError, match="fit"):
        chain.expected_regime_duration("A")