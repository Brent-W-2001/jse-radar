"""
Regime Predictor — logistic regression predicting whether a stock will
outperform the ALSI over the next 21 trading days, conditioned on its
current signals and macro regime.

What this model answers:
  "Given this stock's current momentum/RSI/trend signals AND the macro
  regime we're in, what's the probability it beats the index over the
  next month?"

What this model does NOT answer:
  - It does not predict absolute returns, only RELATIVE outperformance
    vs the ALSI — this is deliberate. A model trained on absolute
    returns would mostly just learn "markets go up over time" and tell
    you nothing about stock SELECTION.
  - It is not a guarantee or even a strong claim — logistic regression
    is used here deliberately as the simplest, most interpretable model
    that can support this number of genuinely independent observations
    (28 tickers, ~140 months of macro variation). A more complex model
    would very likely overfit noise rather than find more real signal,
    the same lesson the regime notebook already taught about model/
    sample-size mismatches.

Critical correctness property:
  The train/test split MUST be by date, never by random row shuffling.
  Shuffling rows before splitting would let the model train on data from
  AFTER its test period, which is look-ahead bias in a different guise
  to the one already fixed in master_builder.py — see
  tests/test_regime_predictor.py::test_split_is_strictly_chronological
  for the explicit guard against this.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from jse_radar.config import PROC_MASTER_DIR
from jse_radar.utils.logger import get_logger

logger = get_logger(__name__)

FEATURE_COLUMNS = ["momentum_score", "rsi_14", "trend_signal"]
FORWARD_WINDOW  = 21          # trading days ahead the label looks
ALSI_TICKER     = "^J203.JO"  # the index every stock is measured against


def build_outperformance_label(
    df: pd.DataFrame, forward_window: int = FORWARD_WINDOW
) -> pd.DataFrame:
    """
    Adds two columns to df:
      - fwd_return_21d:        the stock's own forward 21-day return
      - alsi_fwd_return_21d:   the ALSI's forward 21-day return on the
                                SAME date (broadcast to every ticker)
      - outperformed:          1 if fwd_return_21d > alsi_fwd_return_21d,
                                else 0

    df must already contain 'date', 'ticker', and 'close'.
    """
    df = df.copy().sort_values(["ticker", "date"]).reset_index(drop=True)

    df["fwd_return_21d"] = (
        df.groupby("ticker")["close"]
        .transform(lambda s: s.shift(-forward_window) / s - 1)
    )

    alsi = (
        df[df["ticker"] == ALSI_TICKER][["date", "fwd_return_21d"]]
        .rename(columns={"fwd_return_21d": "alsi_fwd_return_21d"})
    )

    df = df.merge(alsi, on="date", how="left")
    df["outperformed"] = (
        df["fwd_return_21d"] > df["alsi_fwd_return_21d"]
    ).astype(int)

    # Where either return is missing (e.g. near the end of the dataset,
    # not enough future days to compute a forward return), the label
    # is meaningless — mark it NaN rather than a fabricated 0 or 1.
    missing_mask = df["fwd_return_21d"].isna() | df["alsi_fwd_return_21d"].isna()
    df.loc[missing_mask, "outperformed"] = np.nan

    return df


def chronological_train_test_split(
    df: pd.DataFrame, test_start_date: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits by DATE, not by randomly shuffled rows. Every row in train
    has a date strictly before test_start_date; every row in test has
    a date on or after it. This is the single most important
    correctness property of this module — see module docstring.
    """
    cutoff = pd.Timestamp(test_start_date)
    train = df[df["date"] < cutoff].copy()
    test  = df[df["date"] >= cutoff].copy()
    return train, test


class RegimePredictor:
    """Logistic regression: P(stock outperforms ALSI | signals, regime)."""

    def __init__(self, feature_columns: list[str] | None = None) -> None:
        self.feature_columns = feature_columns or FEATURE_COLUMNS
        self.regime_columns: list[str] = []   # filled in during fit()
        self.model: LogisticRegression | None = None
        self.scaler: StandardScaler | None = None

    def _prepare_features(
        self, df: pd.DataFrame, fit_encoder: bool = False
    ) -> pd.DataFrame:
        """
        Builds the final feature matrix: numeric signal columns plus
        one-hot encoded regime columns.

        fit_encoder=True (training data) establishes which regime
        columns exist. fit_encoder=False (test/inference data) reuses
        exactly those same columns, adding any missing ones as all-zero
        — this guarantees train and test always have an IDENTICAL set
        of columns in an IDENTICAL order, which sklearn requires.
        """
        numeric = df[self.feature_columns].copy()

        regime_dummies = pd.get_dummies(
            df["composite_regime"], prefix="regime"
        )

        if fit_encoder:
            self.regime_columns = sorted(regime_dummies.columns.tolist())

        # Reindex to the FIXED set of regime columns established at fit
        # time — adds zero-columns for regimes absent in this slice,
        # drops any regime present here but unseen during training
        regime_dummies = regime_dummies.reindex(
            columns=self.regime_columns, fill_value=0
        )

        return pd.concat([numeric, regime_dummies], axis=1)

    def fit(self, train_df: pd.DataFrame) -> "RegimePredictor":
        """
        Fit the logistic regression on training data.

        train_df must contain: date, ticker, the feature columns,
        composite_regime, and outperformed (the label). Rows with a
        NaN label or NaN features are dropped — logistic regression
        cannot use them, and silently imputing a fabricated value
        would distort what the model actually learns.
        """
        usable = train_df.dropna(
            subset=self.feature_columns + ["composite_regime", "outperformed"]
        )
        dropped = len(train_df) - len(usable)
        if dropped:
            logger.info(f"Dropped {dropped:,} rows with missing features/label")

        X = self._prepare_features(usable, fit_encoder=True)
        y = usable["outperformed"].astype(int)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X_scaled, y)

        logger.info(
            f"Fitted RegimePredictor on {len(usable):,} rows, "
            f"{len(self.regime_columns)} regime categories, "
            f"base rate (outperformed=1): {y.mean():.3f}"
        )
        return self

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        """
        Returns the predicted probability of outperformance for each
        row in df. Rows with missing features/regime produce NaN rather
        than a fabricated probability.
        """
        if self.model is None:
            raise RuntimeError("Call fit() before predict_proba().")

        usable_mask = df[self.feature_columns + ["composite_regime"]].notna().all(axis=1)

        result = pd.Series(np.nan, index=df.index)
        if usable_mask.any():
            X = self._prepare_features(df.loc[usable_mask], fit_encoder=False)
            X_scaled = self.scaler.transform(X)
            probs = self.model.predict_proba(X_scaled)[:, 1]  # P(class=1)
            result.loc[usable_mask] = probs

        return result

    def coefficient_summary(self) -> pd.DataFrame:
        """
        Returns each feature's fitted coefficient, sorted by absolute
        magnitude — the most directly interpretable output of a
        logistic regression. A positive coefficient means higher values
        of that feature push the prediction toward "outperform";
        negative means the opposite.
        """
        if self.model is None:
            raise RuntimeError("Call fit() before requesting coefficients.")

        feature_names = self.feature_columns + self.regime_columns
        coefs = self.model.coef_[0]

        summary = pd.DataFrame({
            "feature": feature_names,
            "coefficient": coefs,
        })
        summary["abs_coefficient"] = summary["coefficient"].abs()
        return summary.sort_values("abs_coefficient", ascending=False).drop(
            columns="abs_coefficient"
        ).reset_index(drop=True)