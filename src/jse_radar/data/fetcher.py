"""
Base class and utilities for fetching market and macro data.
Concrete implementations will subclass DataFetcher.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DataFetcher(ABC):
    """Abstract base for all data fetchers."""

    def __init__(self, raw_dir: Path) -> None:
        self.raw_dir = raw_dir
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def fetch(self, **kwargs) -> None:
        """Download data and save to raw_dir."""
        ...

    @abstractmethod
    def load(self, **kwargs):
        """Load previously fetched data and return a DataFrame."""
        ...