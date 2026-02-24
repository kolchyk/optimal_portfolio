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
# Strategy parameters (fixed — no optimization)
# ---------------------------------------------------------------------------
KAMA_PERIOD: int = 20  # 1 trading month
LOOKBACK_PERIOD: int = 60  # 1 quarter momentum
TOP_N: int = 20  # 20 stocks, ~5% each
KAMA_BUFFER: float = 0.01  # 1% hysteresis buffer

# ---------------------------------------------------------------------------
# Tickers
# ---------------------------------------------------------------------------
SPY_TICKER: str = "SPY"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_DIR: Path = Path("output")
CACHE_DIR: Path = DEFAULT_OUTPUT_DIR / "cache"
