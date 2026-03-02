"""Strategy parameters — hybrid R² Momentum + vol-targeting."""

from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyParams:
    """Immutable parameter set for the hybrid strategy.

    Combines R² Momentum scoring (Clenow) with vol-targeting overlay.
    Uses ATR risk parity for position sizing and hybrid rebalancing:
    periodic rotation + daily KAMA-break exits + daily vol-targeting trim.
    """

    # --- R² Momentum scoring ---
    r2_window: int = 30
    """OLS regression lookback window (days) for R² momentum scoring."""

    kama_asset_period: int = 13
    """KAMA period for individual asset trend filter."""

    kama_buffer: float = 0.04
    """Hysteresis buffer for KAMA filters."""

    # --- ATR risk parity sizing ---
    atr_period: int = 20
    """ATR lookback for position sizing."""

    risk_factor: float = 0.001
    """Risk per position per day (Clenow default)."""

    # --- Portfolio construction ---
    top_n: int = 10
    """Number of positions to hold."""

    rebal_days: int = 18
    """Periodic rotation check frequency (trading days)."""

    max_per_class: int = 5
    """Maximum positions from the same asset class (uses ASSET_CLASS_MAP)."""

    # --- Vol-targeting overlay ---
    target_vol: float = 0.12
    """Target annualised portfolio volatility (e.g. 0.10 = 10%)."""

    max_leverage: float = 1.1
    """Maximum scale factor — caps exposure in calm markets."""

    portfolio_vol_lookback: int = 25
    """Trailing trading days for realised portfolio vol estimation."""

    def __post_init__(self):
        if self.r2_window <= 0:
            raise ValueError(f"r2_window must be positive, got {self.r2_window}")

    @property
    def warmup(self) -> int:
        """Minimum bars needed before trading can start."""
        return max(
            self.r2_window,
            self.kama_asset_period,
            self.atr_period,
            self.portfolio_vol_lookback,
        ) + 10
