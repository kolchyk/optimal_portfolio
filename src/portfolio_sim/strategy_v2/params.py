"""V2 strategy parameters — extends base params with volatility targeting."""

from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyParamsV2:
    """Immutable parameter set for the vol-targeted KAMA momentum strategy.

    Adds portfolio-level volatility targeting on top of the base KAMA
    momentum parameters.  The overlay scales position sizes inversely
    to realised portfolio volatility, keeping annualised vol near
    *target_vol*.
    """

    # --- Base KAMA momentum params (same fields as StrategyParams) ---
    kama_period: int = 20
    kama_spy_period: int = 40
    lookback_period: int = 60
    top_n: int = 6
    kama_buffer: float = 0.01
    use_risk_adjusted: bool = True
    volatility_lookback: int = 20
    oos_days: int = 21
    corr_threshold: float = 0.7
    weighting_mode: str = "risk_parity"

    # --- Vol-targeting overlay (new) ---
    target_vol: float = 0.10
    """Target annualised portfolio volatility (e.g. 0.10 = 10%)."""

    max_leverage: float = 1.5
    """Maximum scale factor — caps exposure in calm markets."""

    portfolio_vol_lookback: int = 21
    """Trailing trading days for realised portfolio vol estimation."""

    @property
    def warmup(self) -> int:
        """Minimum bars needed before trading can start."""
        return max(
            self.lookback_period,
            self.kama_period,
            self.kama_spy_period,
            self.volatility_lookback,
            self.portfolio_vol_lookback,
        ) + 10
