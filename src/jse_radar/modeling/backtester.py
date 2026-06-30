"""
Backtester — walk-forward evaluation of RegimePredictor against a
buy-and-hold ALSI baseline, including realistic rebalancing costs.

What this answers:
  "If I had actually followed this model's top picks every month,
  historically, would I have beaten just holding the index — after
  accounting for the cost of rebalancing?"

What this does NOT answer:
  - It does not guarantee future performance. Every number here
    describes what WOULD HAVE happened historically, on a sample
    that — exactly like the regime analysis — may be too short to
    draw strong conclusions from for any individual period.
  - It does not model real-world frictions beyond a simple per-
    rebalance cost haircut: no market impact, no partial fills, no
    liquidity constraints specific to smaller JSE names. The reported
    numbers are a reasonable approximation, not a precise simulation.

Walk-forward design:
  At each monthly rebalance date, RegimePredictor is refit on ALL data
  strictly before that date (reusing chronological_train_test_split),
  then used to rank stocks by predicted P(outperform) for that date.
  The top N stocks are "held" for the next 21 trading days, equally
  weighted, and their ACTUAL realised return is compared to the ALSI's
  actual realised return over the same window. This is repeated across
  every available monthly cutoff in the data, never reusing a single
  train/test split — this is what makes it "walk-forward" rather than
  a single potentially-lucky split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from jse_radar.modeling.regime_predictor import (
    RegimePredictor,
    build_outperformance_label,
    chronological_train_test_split,
    ALSI_TICKER,
)
from jse_radar.config import PROC_MASTER_DIR
from jse_radar.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_TOP_N = 5
DEFAULT_REBALANCE_COST_BPS = 15   # 0.15% per rebalance, a conservative
                                   # round-trip estimate for liquid JSE names
MIN_TRAIN_MONTHS_REQUIRED = 12    # don't even attempt a walk-forward step
                                   # until there's at least a year of
                                   # training history behind it


def generate_monthly_rebalance_dates(df: pd.DataFrame, min_train_months: int = MIN_TRAIN_MONTHS_REQUIRED) -> list[pd.Timestamp]:
    """
    Returns the first trading date of every month in df, skipping the
    first `min_train_months` months entirely — there's no point
    evaluating a model trained on, say, 2 months of history, since
    we already know from the regime notebook that conclusions drawn
    from thin samples aren't trustworthy.
    """
    df = df.copy()
    df["year_month"] = df["date"].dt.to_period("M")
    monthly_first_dates = (
        df.groupby("year_month")["date"].min().sort_values()
    )

    if len(monthly_first_dates) <= min_train_months:
        return []

    return monthly_first_dates.iloc[min_train_months:].tolist()


def run_walk_forward_backtest(
    df: pd.DataFrame,
    top_n: int = DEFAULT_TOP_N,
    rebalance_cost_bps: float = DEFAULT_REBALANCE_COST_BPS,
    min_train_months: int = MIN_TRAIN_MONTHS_REQUIRED,
) -> pd.DataFrame:
    """
    Run the full walk-forward backtest and return a results DataFrame,
    one row per rebalance date, with columns:
      rebalance_date, picked_tickers, portfolio_return, alsi_return,
      excess_return, net_excess_return (after cost), beat_alsi

    df must already contain the outperformance label columns (run
    build_outperformance_label() first) plus the feature columns and
    composite_regime.
    """
    rebalance_dates = generate_monthly_rebalance_dates(df, min_train_months)
    if not rebalance_dates:
        logger.warning(
            f"Not enough history for a single walk-forward step "
            f"(need at least {min_train_months} months). Returning empty results."
        )
        return pd.DataFrame()

    results = []
    cost_fraction = rebalance_cost_bps / 10_000  # bps -> decimal

    for rebalance_date in rebalance_dates:
        train, test_window = chronological_train_test_split(
            df, test_start_date=rebalance_date.strftime("%Y-%m-%d")
        )

        # The model needs a clean, labelled training set
        train_usable = train.dropna(subset=["outperformed"])
        if train_usable.empty or train_usable["outperformed"].nunique() < 2:
            # Can't fit a binary classifier with only one class present
            logger.debug(f"Skipping {rebalance_date.date()} — insufficient class variety in training data")
            continue

        model = RegimePredictor()
        try:
            model.fit(train_usable)
        except Exception as e:
            logger.warning(f"Skipping {rebalance_date.date()} — fit failed: {e}")
            continue

        # Predict on the SINGLE rebalance date's snapshot only — one
        # row per ticker, exactly as it would look "live" on that date
        snapshot = test_window[test_window["date"] == rebalance_date].copy()
        if snapshot.empty:
            continue

        snapshot["predicted_proba"] = model.predict_proba(snapshot)
        snapshot = snapshot.dropna(subset=["predicted_proba", "fwd_return_21d"])
        # Never trade the index itself as a "pick"
        snapshot = snapshot[snapshot["ticker"] != ALSI_TICKER]

        if len(snapshot) < top_n:
            logger.debug(
                f"Skipping {rebalance_date.date()} — only {len(snapshot)} "
                f"usable tickers, fewer than top_n={top_n}"
            )
            continue

        picks = snapshot.nlargest(top_n, "predicted_proba")

        # Equal-weighted realised return of the picks over the next 21d
        portfolio_return = picks["fwd_return_21d"].mean()
        alsi_return       = picks["alsi_fwd_return_21d"].iloc[0]  # same for every row that date

        excess_return     = portfolio_return - alsi_return
        net_excess_return = excess_return - cost_fraction  # one rebalance cost charged this period

        results.append({
            "rebalance_date":     rebalance_date,
            "picked_tickers":     ", ".join(picks["ticker"].tolist()),
            "portfolio_return":   portfolio_return,
            "alsi_return":        alsi_return,
            "excess_return":      excess_return,
            "net_excess_return":  net_excess_return,
            "beat_alsi":          excess_return > 0,
        })

    results_df = pd.DataFrame(results)
    logger.info(
        f"Walk-forward backtest complete: {len(results_df)} rebalance periods evaluated"
    )
    return results_df


def summarise_backtest(results_df: pd.DataFrame) -> dict:
    """
    Turns the per-period results into the honest summary metrics:
    hit rate, average excess return (gross and net of costs), and a
    Sharpe-style consistency ratio. Returns a dict rather than printing
    directly, so callers (notebook, dashboard, CLI) can format it
    however suits them.
    """
    if results_df.empty:
        return {
            "n_periods": 0,
            "note": "No rebalance periods were evaluated — insufficient data.",
        }

    n_periods         = len(results_df)
    hit_rate           = results_df["beat_alsi"].mean()
    avg_excess_gross   = results_df["excess_return"].mean()
    avg_excess_net     = results_df["net_excess_return"].mean()
    std_excess_net     = results_df["net_excess_return"].std()

    # A simple Sharpe-style ratio: mean / std of net excess return per
    # period, annualised by sqrt(12) since each period is ~1 month.
    # Guard against division by zero with too few periods or zero variance.
    if std_excess_net and std_excess_net > 0:
        sharpe_style = (avg_excess_net / std_excess_net) * np.sqrt(12)
    else:
        sharpe_style = float("nan")

    return {
        "n_periods":                  n_periods,
        "hit_rate":                   round(hit_rate, 3),
        "avg_excess_return_gross":    round(avg_excess_gross, 4),
        "avg_excess_return_net":      round(avg_excess_net, 4),
        "sharpe_style_ratio":         round(sharpe_style, 2) if sharpe_style == sharpe_style else None,
        "note": (
            f"Evaluated over {n_periods} monthly rebalance periods. "
            + (
                "Fewer than 24 periods is a thin sample — treat conclusions "
                "with the same caution as the under-sampled macro regimes "
                "in the regime analysis notebook."
                if n_periods < 24 else
                "Reasonable sample size for a directional conclusion, though "
                "still a single historical path, not a guarantee."
            )
        ),
    }