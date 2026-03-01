"""Configuration for hybrid R² Momentum + vol-targeting strategy."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Trading costs
# ---------------------------------------------------------------------------
INITIAL_CAPITAL: float = 10_000
COMMISSION_RATE: float = 0.0002  # 2 bps (0.02%), Interactive Brokers-like
SLIPPAGE_RATE: float = 0.0005  # 5 bps (0.05%)
RISK_FREE_RATE: float = 0.04
MARGIN_SPREAD: float = 0.015   # 1.5% annual spread over RF for margin borrowing

# ---------------------------------------------------------------------------
# Tickers
# ---------------------------------------------------------------------------
SPY_TICKER: str = "SPY"

# ---------------------------------------------------------------------------
# Cross-asset ETF universe (~64 tickers) — 1–2 repr. per sub-category (no ρ≈0.99 dupes)
# ---------------------------------------------------------------------------
ETF_UNIVERSE: list[str] = [
    # US Equity (15 + QQQ)
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "JPM", "LLY",
    "BAC", "WFC", "GS", "ORCL", "NFLX",
    "QQQ",

    # US Sectors (11)
    "XLK", "XLF", "XLE", "XLV", "XLI", "SMH",
    "XLC", "XLU", "XLY", "XLP", "XLB",

    # International (7) — VEA repr. developed ex-US; EFA/VXUS/SCHF removed (duplicates)
    "VEA", "VWO", "ASML", "TSM", "MELI", "EEM", "EWZ",

    # Bonds (11 — different durations)
    "TLT", "IEF", "SHY", "LQD", "HYG", "EMB",
    "BND", "TIP", "VCSH", "IGIB", "BSV",

    # Metals & Commodities (9)
    "GLD", "SLV", "CPER", "LIT",
    "DBC", "USO", "UNG", "COPX", "GDX",

    # Real Estate (5) — VNQ/REM/VNQI + IYR/XLRE (ликвидные, достаточная история для max R² window=120)
    # WTRE/ERET/VRAI/INDS/SRET удалены: малая ликвидность или молодые (ERET с 2022)
    "VNQ", "VNQI", "REM",
    "IYR", "XLRE",

    # Crypto (2) — IBIT (BTC), ETHA (ETH); FBTC/GBTC/ARKB/HODL/BITB removed (ρ≈0.99)
    "IBIT", "ETHA",
]

ASSET_CLASS_MAP: dict[str, str] = {
    # US Equity
    "AAPL": "US Equity", "MSFT": "US Equity", "GOOGL": "US Equity",
    "AMZN": "US Equity", "NVDA": "US Equity", "META": "US Equity",
    "TSLA": "US Equity", "BRK-B": "US Equity", "JPM": "US Equity",
    "LLY": "US Equity", "BAC": "US Equity", "WFC": "US Equity",
    "GS": "US Equity", "ORCL": "US Equity", "NFLX": "US Equity",
    "QQQ": "US Equity",

    # US Sector ETFs
    "XLK": "US Sector ETF", "XLF": "US Sector ETF", "XLE": "US Sector ETF",
    "XLV": "US Sector ETF", "XLI": "US Sector ETF", "SMH": "US Sector ETF",
    "XLC": "US Sector ETF", "XLU": "US Sector ETF", "XLY": "US Sector ETF",
    "XLP": "US Sector ETF", "XLB": "US Sector ETF",

    # International
    "VEA": "Intl ETF", "VWO": "Intl ETF", "EEM": "Intl ETF", "EWZ": "Intl ETF",
    "ASML": "Intl Equity", "TSM": "Intl Equity", "MELI": "Intl Equity",

    # Bonds
    "TLT": "Long Bonds", "IEF": "Mid Bonds", "SHY": "Short Bonds",
    "LQD": "Corporate Bonds", "HYG": "Corporate Bonds", "EMB": "Corporate Bonds",
    "BND": "Corporate Bonds", "TIP": "Corporate Bonds", "VCSH": "Short Bonds",
    "IGIB": "Corporate Bonds", "BSV": "Short Bonds",

    # Metals
    "GLD": "Metals", "SLV": "Metals", "CPER": "Metals", "LIT": "Metals",
    "DBC": "Metals", "USO": "Metals", "UNG": "Metals", "COPX": "Metals", "GDX": "Metals",

    # Real Estate
    "VNQ": "Real Estate", "VNQI": "Real Estate", "REM": "Real Estate",
    "IYR": "Real Estate", "XLRE": "Real Estate",

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
    # R² Momentum params (r2_windows/weights are fixed, not optimized)
    "kama_asset_period": {"type": "categorical", "choices": [10, 20, 30, 40, 50]},
    "kama_buffer": {"type": "float", "low": 0.005, "high": 0.03, "step": 0.005},
    "rebal_period_weeks": {"type": "int", "low": 1, "high": 6, "step": 1},
    "gap_threshold": {"type": "float", "low": 0.10, "high": 0.20, "step": 0.025},
    "max_per_class": {"type": "int", "low": 2, "high": 5, "step": 1},
    # Vol-targeting params
    "portfolio_vol_lookback": {"type": "int", "low": 15, "high": 35, "step": 5},
}

PARAM_NAMES: list[str] = [
    "kama_asset_period", "kama_buffer",
    "rebal_period_weeks", "gap_threshold",
    "max_per_class", "portfolio_vol_lookback",
]

DEFAULT_N_TRIALS: int = 200
MAX_DD_LIMIT: float = 0.35


def get_kama_periods(space: dict[str, dict] | None = None) -> list[int]:
    """Extract all possible KAMA period values from search space."""
    space = space or SEARCH_SPACE
    periods: list[int] = []
    for key in ("kama_asset_period",):
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
    "oos_weeks": {"type": "int", "low": 10, "high": 20, "step": 10},
    "min_is_weeks": {"type": "int", "low": 20, "high": 40, "step": 10},
}
