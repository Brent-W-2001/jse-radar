"""
Markov chain over macro regimes — transition probabilities and
next-regime forecasting.

What this model answers:
  "Given that we are currently in regime X, what is the probability
  of being in each other regime next month?"

What this model does NOT answer:
  - It does not predict equity returns directly (see regime_predictor.py
    for that, built on top of this)
  - It does not account for HOW LONG we've been in the current regime,
    or any other history beyond the single most recent state (the
    "Markov property" — see module docstring discussion in the
    accompanying notebook for why this is a simplification, not a fact)
  - A transition probability estimated from very few historical
    occurrences of the FROM-regime is a description of a small sample,
    not a reliable forecast. We explicitly track and surface this.

How it's estimated:
  Maximum likelihood estimation of a first-order Markov chain: count
  every observed (regime_t, regime_t+1) pair, normalise each row of
  the resulting count matrix to sum to 1. This is the simplest and
  most standard way to estimate transition probabilities and requires
  no additional assumptions beyond the Markov property itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from jse_radar.config import PROC_MACRO_DIR
from jse_radar.utils.logger import get_logger

logger = get_logger(__name__)

# Below this many historical occurrences of the FROM-regime, we flag
# the entire row of transition probabilities as low-confidence — this
# mirrors the MIN_SPELLS discipline from the regime analysis notebook.
MIN_OCCURRENCES_FOR_CONFIDENCE = 10


class RegimeMarkovChain:
    """First-order Markov chain estimated from observed macro regime history."""

    def __init__(self, macro_dir: Path = PROC_MACRO_DIR) -> None:
        self.macro_dir = macro_dir
        self.transition_matrix: pd.DataFrame | None = None
        self.occurrence_counts: pd.Series | None = None
        self.regimes: list[str] = []

    def _latest(self, pattern: str) -> Path:
        files = sorted(self.macro_dir.glob(pattern), key=lambda f: f.stat().st_mtime)
        if not files:
            raise FileNotFoundError(f"No files matching {pattern} in {self.macro_dir}")
        return files[-1]

    def fit(self, regime_series: pd.Series | None = None) -> "RegimeMarkovChain":
        """
        Estimate the transition matrix from observed regime history.

        If regime_series is not provided, loads the most recent
        macro_regimes_*.parquet and uses its composite_regime column.

        regime_series must be ordered chronologically — this is NOT
        re-sorted internally, since the caller's data may have a
        meaningful index (e.g. a DatetimeIndex) we don't want to disturb.
        """
        if regime_series is None:
            path = self._latest("macro_regimes_*.parquet")
            logger.info(f"Loading regime history from {path}")
            df = pd.read_parquet(path, engine="pyarrow")
            df = df.dropna(subset=["composite_regime"]).sort_values("date")
            regime_series = df["composite_regime"].reset_index(drop=True)

        # Every regime that has EVER occurred, in a stable, sorted order —
        # this fixes the row/column ordering of the matrix regardless of
        # which regimes happen to be present in any given fit() call
        self.regimes = sorted(regime_series.unique().tolist())

        # Build (from, to) pairs: each row's regime and the NEXT row's regime
        current_state = regime_series.iloc[:-1].reset_index(drop=True)
        next_state     = regime_series.iloc[1:].reset_index(drop=True)

        # Count every observed transition
        transition_counts = (
            pd.DataFrame({"from": current_state, "to": next_state})
            .groupby(["from", "to"])
            .size()
            .unstack(fill_value=0)
        )

        # Ensure the matrix has every regime as both a row and a column,
        # even ones that were never observed as a FROM or TO state in
        # this particular dataset — keeps the matrix square and complete
        transition_counts = transition_counts.reindex(
            index=self.regimes, columns=self.regimes, fill_value=0
        )

        # How many times did we observe each FROM-regime in total?
        # This is the denominator for normalising each row, and also
        # the basis for the confidence flag.
        self.occurrence_counts = transition_counts.sum(axis=1)

        # Normalise each row to sum to 1 -> probabilities.
        # Where a regime was never observed as a FROM-state at all
        # (occurrence_count == 0), leave the row as NaN rather than
        # dividing by zero or fabricating a uniform distribution —
        # we genuinely have no information about what follows a
        # regime we've never seen.
        with np.errstate(invalid="ignore", divide="ignore"):
            self.transition_matrix = transition_counts.div(
                self.occurrence_counts, axis=0
            )

        logger.info(
            f"Fitted Markov chain on {len(regime_series)} observations, "
            f"{len(self.regimes)} distinct regimes"
        )
        low_confidence = self.occurrence_counts[
            self.occurrence_counts < MIN_OCCURRENCES_FOR_CONFIDENCE
        ]
        if not low_confidence.empty:
            logger.warning(
                f"Low-confidence transition rows (fewer than "
                f"{MIN_OCCURRENCES_FOR_CONFIDENCE} historical occurrences as "
                f"the FROM-regime):\n{low_confidence.to_string()}"
            )

        return self

    def predict_next_regime_probabilities(self, current_regime: str) -> pd.Series:
        """
        Return the probability distribution over next-month regimes,
        given the current regime.

        Raises a clear error if the model hasn't been fit, or if the
        given regime was never observed in the training history.
        """
        if self.transition_matrix is None:
            raise RuntimeError("Call fit() before predicting.")
        if current_regime not in self.transition_matrix.index:
            raise ValueError(
                f"'{current_regime}' was never observed in the training "
                f"history. Known regimes: {self.regimes}"
            )

        row = self.transition_matrix.loc[current_regime]
        if row.isna().all():
            raise ValueError(
                f"'{current_regime}' was observed in the data but never as "
                f"a FROM-state (e.g. it only ever appeared as the final "
                f"observation) — no transition probabilities available."
            )
        return row.sort_values(ascending=False)

    def confidence_for(self, regime: str) -> dict:
        """
        Returns a small report on how much historical evidence backs
        the transition probabilities FROM this specific regime — the
        same discipline as the regime notebook's spell-count filter,
        applied here to transition estimates instead of forward returns.
        """
        if self.occurrence_counts is None:
            raise RuntimeError("Call fit() before checking confidence.")

        count = int(self.occurrence_counts.get(regime, 0))
        return {
            "regime": regime,
            "historical_occurrences": count,
            "reliable": count >= MIN_OCCURRENCES_FOR_CONFIDENCE,
            "note": (
                f"Estimated from {count} historical month(s) in this regime — "
                + (
                    "sufficient sample size."
                    if count >= MIN_OCCURRENCES_FOR_CONFIDENCE
                    else f"BELOW the {MIN_OCCURRENCES_FOR_CONFIDENCE}-occurrence "
                         f"threshold; treat these transition probabilities as "
                         f"indicative only, not reliable."
                )
            ),
        }

    def expected_regime_duration(self, regime: str) -> float:
        """
        Expected number of months a regime persists once entered, derived
        directly from its self-transition probability: for a geometric
        distribution with "success" probability p = P(stay in regime),
        the expected duration is 1 / (1 - p).

        This is a clean, standard Markov-chain result and gives a second,
        independent way to sanity-check the empirical average spell
        lengths already computed by hand in the regime notebook (Cell 5)
        — the two numbers should be reasonably close if the regimes
        genuinely behave in a Markov-consistent way.
        """
        if self.transition_matrix is None:
            raise RuntimeError("Call fit() before computing expected duration.")

        p_stay = self.transition_matrix.loc[regime, regime]
        if pd.isna(p_stay):
            return float("nan")
        if p_stay >= 1.0:
            return float("inf")
        return 1.0 / (1.0 - p_stay)

    def save(self, out_dir: Path | None = None) -> Path:
        """Save the fitted transition matrix to parquet for reuse/inspection."""
        if self.transition_matrix is None:
            raise RuntimeError("Call fit() before saving.")

        out_dir = out_dir or self.macro_dir
        filename = f"markov_transition_matrix_{datetime.now().strftime('%Y%m%d')}.parquet"
        out_path = out_dir / filename
        self.transition_matrix.to_parquet(out_path, engine="pyarrow")
        logger.info(f"Transition matrix saved → {out_path}")
        return out_path