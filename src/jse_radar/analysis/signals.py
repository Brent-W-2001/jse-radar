"""
Signals — momentum and mean reversion signals for JSE equities.

What is a signal?
  A signal is a number that tells you something about the likely
  direction of a stock's future return. It's not a prediction —
  it's a structured observation derived from price history.

We compute three families of signals here:

1. MOMENTUM signals
   The observation that stocks which have gone up recently tend to
   continue going up over the next 1-12 months (Jegadeesh & Titman 1993).
   We measure this as the stock's return over the past N months.

2. MEAN REVERSION signals
   Over very short horizons (days to weeks), the opposite is true —
   stocks that have moved sharply tend to revert. We capture this
   with z-scores of the price relative to its rolling mean.

3. TREND signals
   Simple moving average crossovers — when the 50-day MA crosses
   above the 200-day MA (a "golden cross"), the stock is in an
   uptrend. When it crosses below (a "death cross"), it's in a
   downtrend.

All signals are computed per ticker using groupby so no ticker's
data contaminates another's calculations.

Output: these columns are added to the master DataFrame and saved.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from jse_radar.config import PROC_MASTER_DIR
from jse_radar.utils.logger import get_logger

logger = get_logger(__name__)


class SignalEngine:
    """Computes momentum, mean reversion, and trend signals on the master frame."""

    def __init__(self, master_dir: Path = PROC_MASTER_DIR) -> None:
        self.master_dir = master_dir

    def _latest(self, pattern: str) -> Path:
        files = sorted(self.master_dir.glob(pattern), key=lambda f: f.stat().st_mtime)
        if not files:
            raise FileNotFoundError(f"No files matching {pattern} in {self.master_dir}")
        return files[-1]

    def compute(self) -> pd.DataFrame:
        """
        Load the master frame, compute all signals, save and return.
        """
        logger.info("Computing equity signals...")

        # ── Load master frame ─────────────────────────────────────────────────
        path = self._latest("master_*.parquet")
        df = pd.read_parquet(path, engine="pyarrow")
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

        logger.info(f"Loaded master frame: {df.shape}")

        # ── 1. Momentum signals ───────────────────────────────────────────────
        # We compute returns over 1, 3, 6, and 12 month lookback windows.
        # In trading days: 1M≈21, 3M≈63, 6M≈126, 12M≈252.
        # shift(1) skips the most recent day — this is standard practice to
        # avoid microstructure noise (bid-ask bounce) contaminating the signal.

        momentum_windows = {
            "mom_1m":  21,
            "mom_3m":  63,
            "mom_6m":  126,
            "mom_12m": 252,
        }

        for col, window in momentum_windows.items():
            df[col] = (
                df.groupby("ticker")["close"]
                .transform(lambda s, w=window: s.shift(1).pct_change(w))
            )
            logger.debug(f"  Computed {col} (window={window})")

        # ── 2. Momentum z-score (cross-sectional) ─────────────────────────────
        # Raw momentum values aren't comparable across stocks (a 30% move in
        # Naspers means something different to a 30% move in Pick n Pay).
        # We standardise each signal cross-sectionally: for each date, compute
        # the z-score of each stock's momentum relative to all other stocks.
        # z = (x - mean) / std
        # A z-score of +2 means this stock has stronger 3M momentum than
        # ~97.5% of all other stocks on that date.

        for col in momentum_windows.keys():
            zscore_col = f"{col}_z"
            df[zscore_col] = (
                df.groupby("date")[col]
                .transform(lambda s: (s - s.mean()) / s.std())
            )
            logger.debug(f"  Computed {zscore_col}")

        # ── 3. Mean reversion signal ──────────────────────────────────────────
        # Z-score of the current price relative to its own 20-day rolling mean.
        # If z > 2: price is extended above recent average → potential reversion down
        # If z < -2: price is depressed below recent average → potential reversion up
        # This is sometimes called a "Bollinger Band z-score"

        df["price_z_20d"] = (
            df.groupby("ticker")["close"]
            .transform(lambda s: (
                (s - s.rolling(20, min_periods=10).mean())
                / s.rolling(20, min_periods=10).std()
            ))
        )

        # ── 4. Moving average signals ─────────────────────────────────────────
        # MA50 and MA200 — the most widely followed trend indicators globally.
        df["ma_50"]  = (
            df.groupby("ticker")["close"]
            .transform(lambda s: s.rolling(50,  min_periods=30).mean())
        )
        df["ma_200"] = (
            df.groupby("ticker")["close"]
            .transform(lambda s: s.rolling(200, min_periods=100).mean())
        )

        # MA ratio: how far above/below the 200-day MA is the stock?
        # > 1 means price is above MA200 (uptrend), < 1 means below (downtrend)
        df["ma_ratio_50_200"] = df["ma_50"] / df["ma_200"]

        # ── 5. Trend signal: golden/death cross ───────────────────────────────
        # +1 = MA50 above MA200 (bullish trend)
        # -1 = MA50 below MA200 (bearish trend)
        #  0 = insufficient data
        df["trend_signal"] = np.where(
            df["ma_50"] > df["ma_200"],  1,
            np.where(
                df["ma_50"] < df["ma_200"], -1, 0
            )
        )

        # ── 6. Relative Strength Index (RSI) ─────────────────────────────────
        # RSI measures the speed and magnitude of recent price changes.
        # RSI > 70: overbought (potential sell signal)
        # RSI < 30: oversold (potential buy signal)
        # Standard window is 14 days.

        def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
            delta = series.diff()
            gain  = delta.clip(lower=0)
            loss  = -delta.clip(upper=0)
            # Use exponential moving average for smoothing (Wilder's method)
            avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
            avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
            rs  = avg_gain / avg_loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            return rsi

        df["rsi_14"] = (
            df.groupby("ticker")["close"]
            .transform(compute_rsi)
        )

        # ── 7. Composite momentum score ───────────────────────────────────────
        # Combine the 1M, 3M, 6M z-scores into one number.
        # We weight longer-horizon signals more (they're more persistent).
        # This gives one number per stock per day: higher = stronger momentum.
        df["momentum_score"] = (
            0.20 * df["mom_1m_z"].fillna(0) +
            0.30 * df["mom_3m_z"].fillna(0) +
            0.50 * df["mom_6m_z"].fillna(0)
        )

        # ── Save ──────────────────────────────────────────────────────────────
        out_dir  = self.master_dir
        filename = f"master_signals_{datetime.now().strftime('%Y%m%d')}.parquet"
        out_path = out_dir / filename
        df.to_parquet(out_path, index=False, engine="pyarrow")

        signal_cols = [
            "mom_1m", "mom_3m", "mom_6m", "mom_12m",
            "mom_1m_z", "mom_3m_z", "mom_6m_z", "mom_12m_z",
            "price_z_20d", "ma_50", "ma_200", "ma_ratio_50_200",
            "trend_signal", "rsi_14", "momentum_score",
        ]
        logger.info(
            f"Signals complete. Added {len(signal_cols)} columns. "
            f"Saved → {out_path}"
        )
        return df