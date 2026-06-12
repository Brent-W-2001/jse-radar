"""
Abstract base class for all data fetchers.

Design principle: every concrete fetcher (equity, macro, world bank)
follows the same contract:
  - __init__ receives the directory where raw data will be saved
  - fetch() downloads and saves data to that directory
  - load() reads previously saved data back as a DataFrame

This means:
  - Fetchers don't know about processors (separation of concerns)
  - You can run fetch() once, then run load() many times without
    hitting the network again
  - KNIME can call each fetcher independently as a Python node
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class DataFetcher(ABC):
    """Abstract base for all data fetchers."""

    def __init__(self, raw_dir: Path) -> None:
        # Ensure the target directory exists before any subclass tries to write
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Fetcher initialised. Raw dir: {self.raw_dir}")

    @abstractmethod
    def fetch(self, start_date: str, end_date: str | None = None) -> None:
        """
        Download data for the given date range and save to self.raw_dir.
        end_date defaults to today if None.
        """
        ...

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """
        Load the most recently fetched data from self.raw_dir.
        Returns a clean pandas DataFrame.
        """
        ...

    def _latest_file(self, pattern: str) -> Path | None:
        """
        Helper: find the most recently modified file matching a glob pattern
        in self.raw_dir. Returns None if no files match.

        This is used by load() implementations to find the latest parquet file
        without hardcoding filenames.
        """
        files = sorted(self.raw_dir.glob(pattern), key=lambda f: f.stat().st_mtime)
        return files[-1] if files else None