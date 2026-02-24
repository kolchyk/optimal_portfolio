"""Strategy parameter container for walk-forward optimization."""

from dataclasses import dataclass

from src.portfolio_sim.config import (
    KAMA_BUFFER,
    KAMA_PERIOD,
    LOOKBACK_PERIOD,
    TOP_N,
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

    @property
    def warmup(self) -> int:
        """Minimum bars needed before trading can start."""
        return max(self.lookback_period, self.kama_period) + 10
