"""Configuration for hybrid R² Momentum + vol-targeting strategy."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Trading costs
# ---------------------------------------------------------------------------
INITIAL_CAPITAL: float = 10_000
COMMISSION_RATE: float = 0.0002  # 2 bps (0.02%), Interactive Brokers-like
SLIPPAGE_RATE: float = 0.0005  # 5 bps (0.05%)
RISK_FREE_RATE: float = 0.04

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
# Optimization search space (R² Momentum + vol-targeting parameters)
# ---------------------------------------------------------------------------
SEARCH_SPACE: dict[str, dict] = {
    # R² Momentum params
    "r2_lookback": {"type": "int", "low": 60, "high": 120, "step": 20},
    "kama_asset_period": {"type": "categorical", "choices": [10, 20, 30, 40, 50]},
    "kama_spy_period": {"type": "categorical", "choices": [20, 30, 40, 50]},
    "kama_buffer": {"type": "float", "low": 0.005, "high": 0.03, "step": 0.005},
    "atr_period": {"type": "int", "low": 10, "high": 30, "step": 5},
    "top_n": {"type": "int", "low": 5, "high": 15, "step": 5},
    "rebal_period_weeks": {"type": "int", "low": 2, "high": 4, "step": 1},
    "gap_threshold": {"type": "float", "low": 0.10, "high": 0.20, "step": 0.025},
    "corr_threshold": {"type": "float", "low": 0.5, "high": 1.0, "step": 0.1},
    # Vol-targeting params
    "target_vol": {"type": "float", "low": 0.05, "high": 0.20, "step": 0.05},
    "max_leverage": {"type": "categorical", "choices": [1.0, 1.25, 1.5, 2.0]},
    "portfolio_vol_lookback": {"type": "int", "low": 15, "high": 35, "step": 10},
}

PARAM_NAMES: list[str] = [
    "r2_lookback", "kama_asset_period", "kama_spy_period", "kama_buffer",
    "atr_period", "top_n", "rebal_period_weeks", "gap_threshold",
    "corr_threshold", "target_vol", "max_leverage", "portfolio_vol_lookback",
]

DEFAULT_N_TRIALS: int = 100
MAX_DD_LIMIT: float = 0.25


def get_kama_periods(space: dict[str, dict] | None = None) -> list[int]:
    """Extract all possible KAMA period values from search space."""
    space = space or SEARCH_SPACE
    periods: list[int] = []
    for key in ("kama_asset_period", "kama_spy_period"):
        spec = space.get(key, {})
        if not spec:
            continue
        if spec.get("type") == "categorical":
            periods.extend(spec["choices"])
        else:
            periods.extend(range(
                spec.get("low", 10),
                spec.get("high", 40) + 1,
                spec.get("step", 5),
            ))
    return sorted(set(periods))


# ---------------------------------------------------------------------------
# Schedule search space (WFO window optimization)
# ---------------------------------------------------------------------------
SCHEDULE_SEARCH_SPACE: dict[str, dict] = {
    "oos_weeks": {"type": "int", "low": 10, "high": 30, "step": 5},
    "min_is_weeks": {"type": "int", "low": 10, "high": 40, "step": 5},
}
