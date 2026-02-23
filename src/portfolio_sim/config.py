"""Typed configuration for portfolio simulation.

All constants are non-tunable. Only StrategyParams fields are optimized by Optuna.
"""

from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Non-tunable constants
# ---------------------------------------------------------------------------
INITIAL_CAPITAL: float = 10_000
COMMISSION_RATE: float = 0.001  # 0.1% per trade
SLIPPAGE_RATE: float = 0.0015  # 15 bps market impact on Open auctions
MAX_WEIGHT: float = 0.15  # Maximum single-position weight
BREADTH_THRESHOLD: float = 0.30  # Market breadth "kill switch"
VOL_FLOOR: float = 0.05  # Minimum annualized volatility for weighting
REBALANCE_INTERVAL: int = 21  # Trading days between rebalances
RISK_FREE_RATE: float = 0.04
SAFE_HAVEN_TICKER: str = "SHV"

# Walk-Forward Validation
WFV_TRAIN_DAYS: int = 252  # 1 year
WFV_TEST_DAYS: int = 63  # 1 quarter

# Paths
TICKERS_JSON_PATH: Path = Path("portfolio_tickers.json")
DEFAULT_OUTPUT_DIR: Path = Path("output")
CACHE_DIR: Path = DEFAULT_OUTPUT_DIR / "cache"


# ---------------------------------------------------------------------------
# Tunable parameters (4 total — Occam's Razor)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StrategyParams:
    """The 4 tunable parameters for the Simple Baseline strategy."""

    kama_period: int = 20  # KAMA indicator period
    lookback_period: int = 126  # Momentum lookback (trading days)
    max_correlation: float = 0.70  # Correlation constraint for diversification
    top_n_selection: int = 15  # Max number of positions


# Optuna search space (mirrors StrategyParams fields)
PARAM_SPACE: dict = {
    "kama_period": {"type": "int", "low": 10, "high": 40, "step": 10},
    "lookback_period": {"type": "int", "low": 63, "high": 126, "step": 21},
    "max_correlation": {"type": "float", "low": 0.5, "high": 0.9, "step": 0.1},
    "top_n_selection": {"type": "int", "low": 10, "high": 25, "step": 5},
}
