"""
Central configuration — paths, constants, environment variables.
This is the single source of truth for the entire project.
Everything imports from here; nothing hardcodes paths directly.
"""

from pathlib import Path
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
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
RAW_EQUITY_DIR   = RAW_DIR / "equities"
RAW_MACRO_DIR    = RAW_DIR / "macro"
RAW_WB_DIR       = RAW_DIR / "worldbank"

PROC_EQUITY_DIR  = PROCESSED_DIR / "equities"
PROC_MACRO_DIR   = PROCESSED_DIR / "macro"
PROC_MASTER_DIR  = PROCESSED_DIR / "master"

# ── Report directories ────────────────────────────────────────────────────────
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# ── API keys (loaded from .env) ───────────────────────────────────────────────
FRED_API_KEY = os.getenv("FRED_API_KEY", "")

# ── JSE equity universe ───────────────────────────────────────────────────────
# Yahoo Finance uses the .JO suffix for JSE-listed stocks.
# These are the Top 40 heavy-hitters we care about most.
# Add or remove tickers here — the fetcher reads from this list.
JSE_TICKERS = {
    # Indices
    "^J203.JO": "ALSI",          # FTSE/JSE All-Share Index
    "^J400.JO": "Top40",         # FTSE/JSE Top 40
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
# FRED series IDs for SA data. Each value is a human-readable label
# we'll use as the column name in our processed DataFrames.
FRED_SERIES = {
    # Monetary policy
    "INTDSRZAM193N": "repo_rate",           # SARB repo rate
    "MPSCZAM193N":   "prime_rate",          # SA prime lending rate
    # Inflation
    "ZAFCPIALLMINMEI": "cpi_all",           # CPI all items
    "ZAFCPICORMINMEI": "cpi_core",          # CPI core (excl food & energy)
    # Exchange rates
    "DEXSFUS":       "zar_usd",             # ZAR per USD (daily)
    "DEXUSEU":       "usd_eur",             # USD per EUR (for cross)
    # Growth and activity
    "NGDPRSAXDCZAM": "gdp_constant",        # GDP constant prices
    "XTEXVA01ZAM664S": "exports_value",     # Exports value index
    "XTIMVA01ZAM664S": "imports_value",     # Imports value index
    # Labour
    "LRUNTTTTZAM156S": "unemployment_rate", # Unemployment rate
    # Money supply
    "MYAGM2ZAM189S": "m2_money_supply",     # M2 money supply
}

# ── World Bank series ─────────────────────────────────────────────────────────
# World Bank indicator codes for South Africa (country code ZAF)
WB_INDICATORS = {
    "NY.GDP.MKTP.KD.ZG": "gdp_growth_pct",         # GDP growth (annual %)
    "FP.CPI.TOTL.ZG":    "inflation_cpi_pct",       # Inflation (annual %)
    "SL.UEM.TOTL.ZS":    "unemployment_pct",        # Unemployment (% of labour force)
    "BN.CAB.XOKA.GD.ZS": "current_account_gdp",    # Current account balance (% GDP)
    "GC.DOD.TOTL.GD.ZS": "govt_debt_gdp",           # Government debt (% GDP)
    "NE.EXP.GNFS.ZS":    "exports_gdp",             # Exports (% GDP)
    "NE.IMP.GNFS.ZS":    "imports_gdp",             # Imports (% GDP)
}

# ── Fetch defaults ────────────────────────────────────────────────────────────
DEFAULT_START_DATE = "2000-01-01"   # How far back to fetch historical data
WB_COUNTRY_CODE    = "ZAF"         # ISO3 code for South Africa