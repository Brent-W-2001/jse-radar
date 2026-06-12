"""
Consistent logging setup for the entire jse-radar project.

Why a custom logger module?
Python's logging module is powerful but verbose to configure.
This module gives every other module a one-liner to get a
properly formatted logger that writes to both the console
and a rotating log file.

Usage in any module:
    from jse_radar.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Fetching data...")
"""

import logging
import logging.handlers
from pathlib import Path
from jse_radar.config import ROOT_DIR, LOG_LEVEL

# The log file lives in the repo root under logs/ (gitignored)
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "jse_radar.log"


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger with the given name, configured with:
    - A StreamHandler (prints to terminal)
    - A RotatingFileHandler (writes to logs/jse_radar.log,
      rotates at 5MB, keeps 3 backups)

    The name is typically __name__ from the calling module,
    which gives log lines like:
        2025-01-01 10:00:00 | INFO     | jse_radar.data.equity_fetcher | Fetching SOL.JO
    """
    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times if get_logger is called
    # more than once with the same name (Python caches loggers by name)
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

    # ── Format ────────────────────────────────────────────────────────────────
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console handler ───────────────────────────────────────────────────────
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    # ── File handler (rotating) ───────────────────────────────────────────────
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger