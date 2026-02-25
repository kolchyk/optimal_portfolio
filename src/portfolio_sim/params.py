"""Strategy parameter container for walk-forward optimization."""

from dataclasses import dataclass

from src.portfolio_sim.config import (
    CORRELATION_LOOKBACK,
    CORRELATION_THRESHOLD,
    KAMA_BUFFER,
    KAMA_PERIOD,
    LOOKBACK_PERIOD,
    TOP_N,
    VOLATILITY_LOOKBACK,
)


@dataclass(frozen=True)
class StrategyParams:
    """Immutable parameter set for a single backtest run.

    frozen=True makes it hashable, usable as dict keys and in sets.
    Default values match the current config.py constants.
    """

    kama_period: int = KAMA_PERIOD
    lookback_period: int = LOOKBACK_PERIOD
    top_n: int = TOP_N
    kama_buffer: float = KAMA_BUFFER
    use_risk_adjusted: bool = False

    # Market regime filter (SPY-based global kill switch)
    enable_regime_filter: bool = True

    # Correlation filter (greedy diversification)
    enable_correlation_filter: bool = False
    correlation_threshold: float = CORRELATION_THRESHOLD
    correlation_lookback: int = CORRELATION_LOOKBACK

    # Position sizing: "equal_weight" or "risk_parity"
    sizing_mode: str = "equal_weight"
    volatility_lookback: int = VOLATILITY_LOOKBACK

    # Max weight per position (1.0 = no cap)
    max_weight: float = 1.0

    @property
    def warmup(self) -> int:
        """Minimum bars needed before trading can start."""
        return max(
            self.lookback_period,
            self.kama_period,
            self.correlation_lookback,
            self.volatility_lookback,
        ) + 10
