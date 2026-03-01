"""Parameter container for Markowitz MVO benchmark strategy."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MVOParams:
    """Immutable parameters for the MVO benchmark backtest.

    frozen=True makes it hashable, usable as dict keys and in sets.
    """

    cov_lookback: int = 252
    min_history: int = 126
    rebal_freq: str = "month"
    max_weight: float = 0.20
    objective: str = "min_variance"
    return_lookback: int = 252

    @property
    def warmup(self) -> int:
        return max(self.cov_lookback, self.return_lookback) + 10
