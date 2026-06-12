"""
Central configuration — paths, constants, environment variables.
This is the single source of truth for the entire project.
Everything imports from here; nothing hardcodes paths directly.
"""

from pathlib import Path
import os

try:
    from dotenv import load_dotenv
    # Pass the explicit path so it works regardless of working directory
    _env_path = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(_env_path)
except ImportError:
    pass

# ── Repo root ─────────────────────────────────────────────────────────────────
# __file__ is src/jse_radar/config.py
# .parents[0] = src/jse_radar/
# .parents[1] = src/
# .parents[2] = jse-radar/   ← repo root
ROOT_DIR = Path(__file__).resolve().parents[2]

# ── Data directories ──────────────────────────────────────────────────────────
DATA_DIR      = ROOT_DIR / "data"
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR  = DATA_DIR / "external"

# Sub-directories created at runtime by fetchers/processors
RAW_EQUITY_DIR  = RAW_DIR / "equities"
RAW_MACRO_DIR   = RAW_DIR / "macro"
RAW_WB_DIR      = RAW_DIR / "worldbank"

PROC_EQUITY_DIR = PROCESSED_DIR / "equities"
PROC_MACRO_DIR  = PROCESSED_DIR / "macro"
PROC_MASTER_DIR = PROCESSED_DIR / "master"

# ── Report directories ────────────────────────────────────────────────────────
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# ── API keys (loaded from .env) ───────────────────────────────────────────────
FRED_API_KEY = os.getenv("FRED_API_KEY", "")

# ── Environment ───────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ── JSE equity universe ───────────────────────────────────────────────────────
JSE_TICKERS = {
    # Indices
    "^J203.JO": "ALSI",
    "^J400.JO": "Top40",
    # Resources
    "AGL.JO":   "Anglo American",
    "BHP.JO":   "BHP Group",
    "SOL.JO":   "Sasol",
    "IMP.JO":   "Impala Platinum",
    "AMS.JO":   "Anglo American Platinum",
    "GFI.JO":   "Gold Fields",
    "HAR.JO":   "Harmony Gold",
    # Financials
    "SBK.JO":   "Standard Bank",
    "FSR.JO":   "FirstRand",
    "NED.JO":   "Nedbank",
    "ABG.JO":   "Absa Group",
    "DSY.JO":   "Discovery",
    "SLM.JO":   "Sanlam",
    "RMI.JO":   "RMI Holdings",
    # Industrials / Consumer
    "NPN.JO":   "Naspers",
    "PRX.JO":   "Prosus",
    "VOD.JO":   "Vodacom",
    "MTN.JO":   "MTN Group",
    "BTI.JO":   "BAT",
    "APN.JO":   "Aspen Pharmacare",
    "BVT.JO":   "Bidvest",
    "TBS.JO":   "Tiger Brands",
    "SPP.JO":   "The Spar Group",
    "PIK.JO":   "Pick n Pay",
    "WHL.JO":   "Woolworths",
    "TRU.JO":   "Truworths",
    "MRP.JO":   "Mr Price",
    "RDF.JO":   "Redefine Properties",
    "GRT.JO":   "Growthpoint",
}

# ── FRED series for South African macro ───────────────────────────────────────
FRED_SERIES = {
    "INTDSRZAM193N":    "repo_rate",
    "MPSCZAM193N":      "prime_rate",
    "ZAFCPIALLMINMEI":  "cpi_all",
    "ZAFCPICORMINMEI":  "cpi_core",
    "DEXSFUS":          "zar_usd",
    "DEXUSEU":          "usd_eur",
    "NGDPRSAXDCZAM":    "gdp_constant",
    "XTEXVA01ZAM664S":  "exports_value",
    "XTIMVA01ZAM664S":  "imports_value",
    "LRUNTTTTZAM156S":  "unemployment_rate",
    "MYAGM2ZAM189S":    "m2_money_supply",
}

# ── World Bank series ─────────────────────────────────────────────────────────
WB_INDICATORS = {
    "NY.GDP.MKTP.KD.ZG": "gdp_growth_pct",
    "FP.CPI.TOTL.ZG":    "inflation_cpi_pct",
    "SL.UEM.TOTL.ZS":    "unemployment_pct",
    "BN.CAB.XOKA.GD.ZS": "current_account_gdp",
    "GC.DOD.TOTL.GD.ZS": "govt_debt_gdp",
    "NE.EXP.GNFS.ZS":    "exports_gdp",
    "NE.IMP.GNFS.ZS":    "imports_gdp",
}

# ── Fetch defaults ────────────────────────────────────────────────────────────
DEFAULT_START_DATE = "2000-01-01"
WB_COUNTRY_CODE    = "ZAF"