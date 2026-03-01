"""Fixed configuration for R² Momentum strategy (Clenow-style).

All parameters are fixed — no optimization, no tuning.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Trading costs
# ---------------------------------------------------------------------------
INITIAL_CAPITAL: float = 10_000
COMMISSION_RATE: float = 0.0002  # 2 bps (0.02%), Interactive Brokers-like
SLIPPAGE_RATE: float = 0.0005  # 5 bps (0.05%)
RISK_FREE_RATE: float = 0.04

# ---------------------------------------------------------------------------
# R² Momentum strategy parameters (Clenow-style)
# ---------------------------------------------------------------------------
KAMA_PERIOD: int = 40       # KAMA period for individual asset trend filter
KAMA_SPY_PERIOD: int = 40   # KAMA period for SPY regime filter
TOP_N: int = 10
KAMA_BUFFER: float = 0.01   # hysteresis buffer for KAMA filters
R2_LOOKBACK: int = 90       # OLS regression lookback (Clenow standard)
GAP_THRESHOLD: float = 0.15 # exclude assets with >15% single-day gap
ATR_PERIOD: int = 20        # ATR lookback for position sizing
RISK_FACTOR: float = 0.001  # risk per position per day (Clenow default)
REBAL_PERIOD_WEEKS: int = 2 # rebalance check every N weeks
OOS_DAYS: int = 21

# ---------------------------------------------------------------------------
# Tickers
# ---------------------------------------------------------------------------
SPY_TICKER: str = "SPY"

# ---------------------------------------------------------------------------
# Cross-asset ETF universe (lean ~37 tickers)
# ---------------------------------------------------------------------------
ETF_UNIVERSE: list[str] = [
    # US Equity (10 + QQQ)
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "JPM", "LLY",
    "QQQ",

    # US Sectors (6)
    "XLK", "XLF", "XLE", "XLV", "XLI", "SMH",

    # International (5)
    "VEA", "VWO", "ASML", "TSM", "MELI",

    # Bonds (6 — different durations)
    "TLT", "IEF", "SHY", "LQD", "HYG", "EMB",

    # Metals & Commodities (4)
    "GLD", "SLV", "CPER", "LIT",

    # Real Estate (3)
    "VNQ", "VNQI", "REM",

    # Crypto (2)
    "IBIT", "ETHA",
]

ASSET_CLASS_MAP: dict[str, str] = {
    # US Equity
    "AAPL": "US Equity", "MSFT": "US Equity", "GOOGL": "US Equity",
    "AMZN": "US Equity", "NVDA": "US Equity", "META": "US Equity",
    "TSLA": "US Equity", "BRK-B": "US Equity", "JPM": "US Equity",
    "LLY": "US Equity", "QQQ": "US Equity",

    # US Sector ETFs
    "XLK": "US Sector ETF", "XLF": "US Sector ETF", "XLE": "US Sector ETF",
    "XLV": "US Sector ETF", "XLI": "US Sector ETF", "SMH": "US Sector ETF",

    # International
    "VEA": "Intl ETF", "VWO": "Intl ETF",
    "ASML": "Intl Equity", "TSM": "Intl Equity", "MELI": "Intl Equity",

    # Bonds
    "TLT": "Long Bonds", "IEF": "Mid Bonds", "SHY": "Short Bonds",
    "LQD": "Corporate Bonds", "HYG": "Corporate Bonds", "EMB": "Corporate Bonds",

    # Metals
    "GLD": "Metals", "SLV": "Metals", "CPER": "Metals", "LIT": "Metals", "PPLT": "Metals",

    # Real Estate
    "VNQ": "Real Estate", "VNQI": "Real Estate", "REM": "Real Estate",

    # Crypto
    "IBIT": "Crypto", "ETHA": "Crypto",
}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_DIR: Path = Path("output")
CACHE_DIR: Path = DEFAULT_OUTPUT_DIR / "cache"


# ---------------------------------------------------------------------------
# Optimization search spaces (R² Momentum)
# ---------------------------------------------------------------------------
R2_SEARCH_SPACE: dict[str, dict] = {
    "r2_lookback": {"type": "int", "low": 60, "high": 120, "step": 20},
    "kama_asset_period": {"type": "categorical", "choices": [10, 20, 30, 40, 50]},
    "kama_spy_period": {"type": "categorical", "choices": [20, 30, 40, 50]},
    "kama_buffer": {"type": "float", "low": 0.005, "high": 0.03, "step": 0.005},
    "gap_threshold": {"type": "float", "low": 0.10, "high": 0.20, "step": 0.025},
    "atr_period": {"type": "int", "low": 10, "high": 30, "step": 5},
    "top_n": {"type": "int", "low": 5, "high": 15, "step": 5},
    "rebal_period_weeks": {"type": "int", "low": 2, "high": 4, "step": 1},
}

DEFAULT_N_TRIALS: int = 50

# ---------------------------------------------------------------------------
# Schedule search space (WFO window optimization)
# ---------------------------------------------------------------------------
SCHEDULE_SEARCH_SPACE: dict[str, dict] = {
    "oos_weeks": {"type": "int", "low": 2, "high": 20, "step": 2},
    "min_is_weeks": {"type": "int", "low": 2, "high": 20, "step": 2},
}
