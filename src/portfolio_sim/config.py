"""Fixed configuration for simplified KAMA momentum strategy.

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
# Strategy parameters — concentrated momentum (4x S&P 500 target)
# ---------------------------------------------------------------------------
KAMA_PERIOD: int = 20
LOOKBACK_PERIOD: int = 60  # Центр надежного плато
TOP_N: int = 10
KAMA_BUFFER: float = 0.024  # Отступили от пика 0.025 на широкое плечо

# ---------------------------------------------------------------------------
# Tickers
# ---------------------------------------------------------------------------
SPY_TICKER: str = "SPY"

# ---------------------------------------------------------------------------
# Cross-asset ETF universe (all-weather tactical allocation)
# ---------------------------------------------------------------------------
ETF_UNIVERSE: list[str] = [
    # US Equities
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "V", "JPM",
    "LLY", "UNH", "MA", "HD", "PG", "XOM", "BAC", "COST", "NFLX", "DIS",
    "KO", "PEP", "NKE", "MCD", "TMO", "ABT", "CRM", "WMT", "WFC", "CMCSA",
    "MRK", "PFE", "CVS", "ALL", "HPQ", "PYPL", "CHTR", "ZS", "HUBS", "PLTR",
    "AMD", "MU", "WDC", "STX", "LRCX", "AMAT", "KLAC", "ADBE", "SHOP", "IREN",

    # International Equities (ADRs)
    "SHEL", "SAP", "ASML", "TSM", "HDB", "SNY", "RYCEY", "AIQUY", "SMFG", "ENB",
    "IBDRY", "SMEGF", "SPOT", "ACN", "ESLOY", "PROSY", "BACHY",

    # Emerging Market Equities (ADRs)
    "BABA", "PDD", "JD", "BIDU", "NTES", "VALE", "MELI", "IBN", "DLO", "GRAB",
    "ARCO", "ABEV", "YUMC", "CPNG", "SEA", "TME", "LI", "NIO", "BEKE", "EDU",

    # US Sector & Thematic ETFs
     "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLU", "XLY", "XLP",
    "XLC", "XLB", "XLRE", "ITA", "VFH", "XRT", "VGT", "XLA", "XAR", "IBB", "SMH", "KBE",

    # International & Regional ETFs
    "VEA", "VXUS", "IEFA", "SCHF", "VPL", "BBEU", "VGK", "EWG", "EWJ", "EWQ",
    "EEM", "VWO", "IEMG", "ILF", "AIA", "EEMA", "FLEU", "SCHY", "VIGI", "AVEM",

    # Bonds & Fixed Income
    "TLT", "IEF", "SHY", "SGOV", "SHV", "LQD", "VCIT", "VTC", "HYG", "JNK",
    "USHY", "FBND", "FIGB", "JCPB", "EMB", "PCY", "BWX", "BSJO", "TLH", "IGIB",

    # Metals (Pure ETFs)
    "GLD", "SLV", "PPLT", "PALL", "CPER", "LIT", "JJU",

    # Real Estate & REITs
    "VNQ", "SCHH", "RWR", "USRT", "REZ", "FRI", "AREA", "RWO", "VNQI", "BBRE",
    "REM", "SRVR", "MORT", "KBWY", "REET", "DFAR", "BCREX", "CSCIX", "PSTL", "ILPT",

    # Crypto ETFs
    "IBIT", "FBTC", "GBTC", "ARKB", "ETHA",
]

ASSET_CLASS_MAP: dict[str, str] = {
    # US Equities
    "AAPL": "US Equity", "MSFT": "US Equity", "GOOGL": "US Equity", "AMZN": "US Equity", "NVDA": "US Equity",
    "META": "US Equity", "TSLA": "US Equity", "BRK-B": "US Equity", "V": "US Equity", "JPM": "US Equity",
    "LLY": "US Equity", "UNH": "US Equity", "MA": "US Equity", "HD": "US Equity", "PG": "US Equity",
    "XOM": "US Equity", "BAC": "US Equity", "COST": "US Equity", "NFLX": "US Equity", "DIS": "US Equity",
    "KO": "US Equity", "PEP": "US Equity", "NKE": "US Equity", "MCD": "US Equity", "TMO": "US Equity",
    "ABT": "US Equity", "CRM": "US Equity", "WMT": "US Equity", "WFC": "US Equity", "CMCSA": "US Equity",
    "MRK": "US Equity", "PFE": "US Equity", "CVS": "US Equity", "ALL": "US Equity", "HPQ": "US Equity",
    "PYPL": "US Equity", "CHTR": "US Equity", "ZS": "US Equity", "HUBS": "US Equity", "PLTR": "US Equity",
    "AMD": "US Equity", "MU": "US Equity", "WDC": "US Equity", "STX": "US Equity", "LRCX": "US Equity",
    "AMAT": "US Equity", "KLAC": "US Equity", "ADBE": "US Equity", "SHOP": "US Equity", "IREN": "US Equity",

    # International Equities
    "SHEL": "Intl Equity", "SAP": "Intl Equity", "ASML": "Intl Equity", "TSM": "Intl Equity", "HDB": "Intl Equity",
    "SNY": "Intl Equity", "RYCEY": "Intl Equity", "AIQUY": "Intl Equity", "SMFG": "Intl Equity", "ENB": "Intl Equity",
    "IBDRY": "Intl Equity", "SMEGF": "Intl Equity", "SPOT": "Intl Equity", "ACN": "Intl Equity", "ESLOY": "Intl Equity",
    "PROSY": "Intl Equity", "BACHY": "Intl Equity",

    # Emerging Market Equities
    "BABA": "EM Equity", "PDD": "EM Equity", "JD": "EM Equity", "BIDU": "EM Equity", "NTES": "EM Equity",
    "VALE": "EM Equity", "MELI": "EM Equity", "IBN": "EM Equity", "DLO": "EM Equity", "GRAB": "EM Equity",
    "ARCO": "EM Equity", "ABEV": "EM Equity", "YUMC": "EM Equity", "CPNG": "EM Equity", "SEA": "EM Equity",
    "TME": "EM Equity", "LI": "EM Equity", "NIO": "EM Equity", "BEKE": "EM Equity", "EDU": "EM Equity",

    # US Sector ETFs
    "QQQ": "US Equity", "XLK": "US Sector ETF", "XLF": "US Sector ETF", "XLE": "US Sector ETF", "XLV": "US Sector ETF", "XLI": "US Sector ETF",
    "XLU": "US Sector ETF", "XLY": "US Sector ETF", "XLP": "US Sector ETF", "XLC": "US Sector ETF", "XLB": "US Sector ETF",
    "XLRE": "US Sector ETF", "ITA": "US Sector ETF", "VFH": "US Sector ETF", "XRT": "US Sector ETF", "VGT": "US Sector ETF",
    "XLA": "US Sector ETF", "XAR": "US Sector ETF", "IBB": "US Sector ETF", "SMH": "US Sector ETF", "KBE": "US Sector ETF",

    # International ETFs
    "VEA": "Intl ETF", "VXUS": "Intl ETF", "IEFA": "Intl ETF", "SCHF": "Intl ETF", "VPL": "Intl ETF",
    "BBEU": "Intl ETF", "VGK": "Intl ETF", "EWG": "Intl ETF", "EWJ": "Intl ETF", "EWQ": "Intl ETF",
    "EEM": "Intl ETF", "VWO": "Intl ETF", "IEMG": "Intl ETF", "ILF": "Intl ETF", "AIA": "Intl ETF",
    "EEMA": "Intl ETF", "FLEU": "Intl ETF", "SCHY": "Intl ETF", "VIGI": "Intl ETF", "AVEM": "Intl ETF",

    # Bonds
    "TLT": "Long Bonds", "IEF": "Mid Bonds", "SHY": "Short Bonds", "SGOV": "Short Bonds", "SHV": "Short Bonds",
    "LQD": "Corporate Bonds", "VCIT": "Corporate Bonds", "VTC": "Corporate Bonds", "HYG": "Corporate Bonds", "JNK": "Corporate Bonds",
    "USHY": "Corporate Bonds", "FBND": "Corporate Bonds", "FIGB": "Corporate Bonds", "JCPB": "Corporate Bonds", "EMB": "Corporate Bonds",
    "PCY": "Corporate Bonds", "BWX": "Corporate Bonds", "BSJO": "Corporate Bonds", "TLH": "Long Bonds", "IGIB": "Corporate Bonds",

    # Metals
    "GLD": "Metals", "SLV": "Metals", "PPLT": "Metals", "PALL": "Metals", "CPER": "Metals", "LIT": "Metals", "JJU": "Metals",

    # Real Estate
    "VNQ": "Real Estate", "SCHH": "Real Estate", "RWR": "Real Estate", "USRT": "Real Estate", "REZ": "Real Estate",
    "FRI": "Real Estate", "AREA": "Real Estate", "RWO": "Real Estate", "VNQI": "Real Estate", "BBRE": "Real Estate",
    "REM": "Real Estate", "SRVR": "Real Estate", "MORT": "Real Estate", "KBWY": "Real Estate", "REET": "Real Estate",
    "DFAR": "Real Estate", "BCREX": "Real Estate", "CSCIX": "Real Estate", "PSTL": "Real Estate", "ILPT": "Real Estate",

    # Crypto
    "BTC-USD": "Crypto", "ETH-USD": "Crypto", "SOL-USD": "Crypto", "BNB-USD": "Crypto", "XRP-USD": "Crypto",
    "ADA-USD": "Crypto", "DOGE-USD": "Crypto", "TRX-USD": "Crypto", "DOT-USD": "Crypto", "LINK-USD": "Crypto",
    "AVAX-USD": "Crypto", "MATIC-USD": "Crypto", "LTC-USD": "Crypto", "BCH-USD": "Crypto", "SHIB-USD": "Crypto",
    "SUI-USD": "Crypto", "NEAR-USD": "Crypto", "APT-USD": "Crypto", "HBAR-USD": "Crypto", "ONDO-USD": "Crypto",
}

# ---------------------------------------------------------------------------
# Correlation filter (greedy diversification)
# ---------------------------------------------------------------------------
CORRELATION_THRESHOLD: float = 0.9

# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------
VOLATILITY_LOOKBACK: int = 20  # trading days for inverse-vol weighting

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_DIR: Path = Path("output")
CACHE_DIR: Path = DEFAULT_OUTPUT_DIR / "cache"


# ---------------------------------------------------------------------------
# Optimization search spaces
# ---------------------------------------------------------------------------
SENSITIVITY_SPACE = {
    "kama_period": {"type": "categorical", "choices": [10, 20, 30, 40]},
    "lookback_period": {"type": "int", "low": 20, "high": 100, "step": 20},
    "kama_buffer": {"type": "float", "low": 0.005, "high": 0.03, "step": 0.05},
    "top_n": {"type": "int", "low": 5, "high": 30, "step": 5},
    "enable_correlation_filter": {"type": "categorical", "choices": [True, False]},
    "correlation_threshold": {"type": "float", "low": 0.6, "high": 0.9, "step": 0.05},
}

MAX_PROFIT_SPACE = {
    "kama_period": {"type": "categorical", "choices": [10, 15, 20, 30]},
    "lookback_period": {"type": "int", "low": 20, "high": 100, "step": 20},
    "top_n": {"type": "int", "low": 5, "high": 30, "step": 5},
    "kama_buffer": {"type": "float", "low": 0.005, "high": 0.03, "step": 0.005},
    "enable_correlation_filter": {"type": "categorical", "choices": [True, False]},
    "correlation_threshold": {"type": "float", "low": 0.5, "high": 0.95, "step": 0.05},
}

DEFAULT_N_TRIALS: int = 50
DEFAULT_MAX_PROFIT_TRIALS: int = 50
