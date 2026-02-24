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
KAMA_PERIOD: int = 10  # Fast adaptive MA (~2 trading weeks)
LOOKBACK_PERIOD: int = 150  # ~6-month momentum window
TOP_N: int = 5  # 5 stocks, ~20% each — concentrated momentum
KAMA_BUFFER: float = 0.008  # 0.8% hysteresis buffer

# ---------------------------------------------------------------------------
# Tickers
# ---------------------------------------------------------------------------
SPY_TICKER: str = "SPY"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_DIR: Path = Path("output")
CACHE_DIR: Path = DEFAULT_OUTPUT_DIR / "cache"
