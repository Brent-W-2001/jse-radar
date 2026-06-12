"""
Pipeline orchestrator — runs all fetchers in the correct order.

This is the module you'd wire into KNIME as a single Python node,
or call from scripts/run_pipeline.py from the terminal.

Design: it doesn't import all three fetchers at the top level.
Instead it imports them inside run() so that a failure in one
fetcher (e.g. missing API key) doesn't block importing the others.
"""

from __future__ import annotations
from jse_radar.utils.logger import get_logger

logger = get_logger(__name__)


def run(
    start_date: str = "2000-01-01",
    end_date: str | None = None,
    fetch_equities: bool = True,
    fetch_macro: bool = True,
    fetch_worldbank: bool = True,
) -> dict[str, bool]:
    """
    Run the full data fetch pipeline.

    Returns a dict of {source: success} so callers can see which
    fetchers succeeded and which failed without an exception stopping
    the whole run.

    Example:
        from jse_radar.data.pipeline import run
        results = run(start_date="2015-01-01")
        # {'equities': True, 'macro': True, 'worldbank': True}
    """
    results = {}

    if fetch_equities:
        try:
            from jse_radar.data.equity_fetcher import EquityFetcher
            fetcher = EquityFetcher()
            fetcher.fetch(start_date=start_date, end_date=end_date)
            results["equities"] = True
        except Exception as e:
            logger.error(f"Equity fetch failed: {e}")
            results["equities"] = False

    if fetch_macro:
        try:
            from jse_radar.data.macro_fetcher import MacroFetcher
            fetcher = MacroFetcher()
            fetcher.fetch(start_date=start_date, end_date=end_date)
            results["macro"] = True
        except Exception as e:
            logger.error(f"Macro fetch failed: {e}")
            results["macro"] = False

    if fetch_worldbank:
        try:
            from jse_radar.data.wb_fetcher import WorldBankFetcher
            fetcher = WorldBankFetcher()
            fetcher.fetch(start_date=start_date, end_date=end_date)
            results["worldbank"] = True
        except Exception as e:
            logger.error(f"World Bank fetch failed: {e}")
            results["worldbank"] = False

    # Summary
    passed = [k for k, v in results.items() if v]
    failed = [k for k, v in results.items() if not v]
    logger.info(f"Pipeline complete. Passed: {passed}. Failed: {failed}.")

    return results