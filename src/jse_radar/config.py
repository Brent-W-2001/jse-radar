"""Central configuration — paths, constants, environment variables."""

from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# ── Repo root (two levels up from src/jse_radar/config.py) ───────────────────
ROOT_DIR = Path(__file__).resolve().parents[2]

# ── Data directories ──────────────────────────────────────────────────────────
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"

# ── Report directories ────────────────────────────────────────────────────────
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# ── Environment ───────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")