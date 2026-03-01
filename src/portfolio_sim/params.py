"""Strategy parameters — hybrid R² Momentum + vol-targeting."""

from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyParams:
    """Immutable parameter set for the hybrid strategy.

    Combines R² Momentum scoring (Clenow) with vol-targeting overlay.
    Uses ATR risk parity for position sizing and hybrid rebalancing:
    periodic rotation + daily KAMA-break exits + daily vol-targeting trim.
    """

    # --- R² Momentum scoring (blended ensemble) ---
    r2_windows: tuple[int, ...] = (60, 90, 120)
    """OLS regression lookback windows for blended R² momentum scoring."""

    r2_weights: tuple[float, ...] = (0.5, 0.3, 0.2)
    """Weights for blending R² scores from each window (must sum to 1.0)."""

    kama_asset_period: int = 20
    """KAMA period for individual asset trend filter."""

    kama_buffer: float = 0.02
    """Hysteresis buffer for KAMA filters."""

    # --- ATR risk parity sizing ---
    atr_period: int = 20
    """ATR lookback for position sizing."""

    risk_factor: float = 0.001
    """Risk per position per day (Clenow default)."""

    # --- Portfolio construction ---
    top_n: int = 10
    """Number of positions to hold."""

    rebal_period_weeks: int = 2
    """Periodic rotation check frequency (weeks)."""

    gap_threshold: float = 0.2
    """Gap exit threshold — exit held positions with |daily_return| > this."""

    max_per_class: int = 3
    """Maximum positions from the same asset class (uses ASSET_CLASS_MAP)."""

    # --- Vol-targeting overlay ---
    target_vol: float = 0.40
    """Target annualised portfolio volatility (e.g. 0.10 = 10%)."""

    max_leverage: float = 2.0
    """Maximum scale factor — caps exposure in calm markets."""

    portfolio_vol_lookback: int = 35
    """Trailing trading days for realised portfolio vol estimation."""

    def __post_init__(self):
        if len(self.r2_windows) != len(self.r2_weights):
            raise ValueError(
                f"r2_windows ({len(self.r2_windows)}) and r2_weights "
                f"({len(self.r2_weights)}) must have the same length"
            )
        if abs(sum(self.r2_weights) - 1.0) > 1e-6:
            raise ValueError(
                f"r2_weights must sum to 1.0, got {sum(self.r2_weights)}"
            )
        if any(w <= 0 for w in self.r2_windows):
            raise ValueError("All r2_windows must be positive")

    @property
    def max_r2_window(self) -> int:
        """Largest R² lookback window (for warmup, correlation, data requirements)."""
        return max(self.r2_windows)

    @property
    def warmup(self) -> int:
        """Minimum bars needed before trading can start."""
        return max(
            self.max_r2_window,
            self.kama_asset_period,
            self.atr_period,
            self.portfolio_vol_lookback,
        ) + 10
