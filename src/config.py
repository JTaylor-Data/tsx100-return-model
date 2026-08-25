"""Shared paths and constants for the TSX-100 return ranking pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DOCS = ROOT / "docs"
MODELS = ROOT / "models"

for _d in (DATA_RAW, DATA_PROCESSED, DOCS, MODELS):
    _d.mkdir(parents=True, exist_ok=True)

# --- Universe ---
UNIVERSE_SIZE = 100
XIC_PRODUCT_ID = "239837"
XIC_HOLDINGS_URL = (
    "https://www.blackrock.com/ca/investors/en/products/239837/"
    "ishares-sptsx-capped-composite-index-etf/1464253357814.ajax"
)
UNIVERSE_FILE = DATA_RAW / "universe.csv"

# --- History / rebalance ---
REBALANCE_FREQ = "MS"          # month-start snapshots
PRICE_HISTORY_YEARS = 15       # how far back to pull OHLCV
FORWARD_RETURN_MONTHS = 12     # label horizon
EMBARGO_MONTHS = 12            # purge/embargo gap for walk-forward CV, = label horizon
REPORTING_LAG_DAYS = 45        # assumed delay before a quarterly filing is public/point-in-time usable

# --- Macro (Bank of Canada Valet API) ---
BOC_SERIES = {
    "boc_overnight_rate": "V39079",
    "yield_2y": "BD.CDN.2YR.DQ.YLD",
    "yield_10y": "BD.CDN.10YR.DQ.YLD",
}
BOC_VALET_URL = "https://www.bankofcanada.ca/valet/observations/{series}/json"
OIL_TICKER = "CL=F"
BENCHMARK_TICKER = "^GSPTSE"  # S&P/TSX Composite index
# BoC's FXUSDCAD series only starts 2017; yfinance's CAD=X goes back to 2003,
# which comfortably covers PRICE_HISTORY_YEARS.
USDCAD_TICKER = "CAD=X"

HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}
