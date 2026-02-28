"""Strategy parameter container for walk-forward optimization."""

from dataclasses import dataclass

from src.portfolio_sim.config import (
    CORR_THRESHOLD,
    KAMA_BUFFER,
    KAMA_PERIOD,
    LOOKBACK_PERIOD,
    OOS_DAYS,
    TOP_N,
    VOLATILITY_LOOKBACK,
    WEIGHTING_MODE,
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
    use_risk_adjusted: bool = True

    # Risk-parity (inverse-volatility) lookback window
    volatility_lookback: int = VOLATILITY_LOOKBACK

    # WFO out-of-sample window (trading days)
    oos_days: int = OOS_DAYS

    # Max pairwise correlation with held positions (1.0 = no filter)
    corr_threshold: float = CORR_THRESHOLD

    # Position sizing mode: "equal_weight" or "risk_parity"
    weighting_mode: str = WEIGHTING_MODE

    @property
    def warmup(self) -> int:
        """Minimum bars needed before trading can start."""
        return max(
            self.lookback_period,
            self.kama_period,
            self.volatility_lookback,
        ) + 10
