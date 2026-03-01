"""S2 strategy parameters — hybrid R² Momentum + vol-targeting."""

from dataclasses import dataclass


@dataclass(frozen=True)
class S2StrategyParams:
    """Immutable parameter set for the S2 hybrid strategy.

    Combines R² Momentum scoring (Clenow) with vol-targeting overlay (V2).
    Uses ATR risk parity for position sizing and hybrid rebalancing:
    periodic rotation + daily KAMA-break exits + daily vol-targeting trim.
    """

    # --- R² Momentum scoring (from R2) ---
    r2_lookback: int = 90
    """OLS regression lookback for R² momentum scoring."""

    kama_asset_period: int = 10
    """KAMA period for individual asset trend filter."""

    kama_spy_period: int = 40
    """KAMA period for SPY regime filter."""

    kama_buffer: float = 0.005
    """Hysteresis buffer for KAMA filters."""

    # --- ATR risk parity sizing (from R2) ---
    atr_period: int = 20
    """ATR lookback for position sizing."""

    risk_factor: float = 0.001
    """Risk per position per day (Clenow default)."""

    # --- Portfolio construction ---
    top_n: int = 5
    """Number of positions to hold."""

    rebal_period_weeks: int = 3
    """Periodic rotation check frequency (weeks)."""

    gap_threshold: float = 0.175
    """Gap exit threshold — exit held positions with |daily_return| > this."""

    corr_threshold: float = 0.7
    """Correlation filter threshold for new entries."""

    # --- Vol-targeting overlay (from V2) ---
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
            self.r2_lookback,
            self.kama_asset_period,
            self.kama_spy_period,
            self.atr_period,
            self.portfolio_vol_lookback,
        ) + 10
