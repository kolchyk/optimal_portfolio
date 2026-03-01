"""Strategy parameter containers for walk-forward optimization."""

from dataclasses import dataclass, fields

from src.portfolio_sim.config import (
    KAMA_BUFFER,
    KAMA_PERIOD,
    KAMA_SPY_PERIOD,
    OOS_DAYS,
    TOP_N,
)


# ---------------------------------------------------------------------------
# V1 KAMA Momentum params (kept for V2 backward compatibility)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StrategyParams:
    """Immutable parameter set for V1/V2 KAMA momentum backtest.

    frozen=True makes it hashable, usable as dict keys and in sets.
    """

    kama_period: int = KAMA_PERIOD
    kama_spy_period: int = KAMA_SPY_PERIOD
    lookback_period: int = 40
    top_n: int = TOP_N
    kama_buffer: float = KAMA_BUFFER
    use_risk_adjusted: bool = True
    volatility_lookback: int = 20
    oos_days: int = OOS_DAYS
    corr_threshold: float = 0.7
    weighting_mode: str = "equal_weight"

    @property
    def warmup(self) -> int:
        return max(
            self.lookback_period,
            self.kama_period,
            self.kama_spy_period,
            self.volatility_lookback,
        ) + 10


# ---------------------------------------------------------------------------
# R² Momentum params (primary strategy)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class R2StrategyParams:
    """Immutable parameter set for R² Momentum backtest."""

    r2_lookback: int = 90
    kama_asset_period: int = KAMA_PERIOD
    kama_spy_period: int = KAMA_SPY_PERIOD
    kama_buffer: float = KAMA_BUFFER
    gap_threshold: float = 0.15
    atr_period: int = 20
    risk_factor: float = 0.001
    top_n: int = TOP_N
    rebal_period_weeks: int = 2

    @property
    def warmup(self) -> int:
        return max(self.r2_lookback, self.kama_asset_period, self.kama_spy_period) + 10


R2_PARAM_NAMES = [f.name for f in fields(R2StrategyParams)]
