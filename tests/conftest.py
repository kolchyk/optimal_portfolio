"""Shared test fixtures for portfolio simulation tests."""

import numpy as np
import pandas as pd
import pytest

from src.portfolio_sim.config import SAFE_HAVEN_TICKER, StrategyParams


@pytest.fixture
def strategy_params() -> StrategyParams:
    """Default strategy parameters."""
    return StrategyParams()


@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    """Generate 500 days of synthetic Close prices for 20 tickers + SHV + SPY.

    Prices follow geometric Brownian motion with varying drift and volatility.
    """
    np.random.seed(42)
    n_days = 500
    tickers = [f"T{i:02d}" for i in range(20)] + [SAFE_HAVEN_TICKER, "SPY"]

    dates = pd.bdate_range("2021-01-04", periods=n_days)
    data = {}

    for i, t in enumerate(tickers):
        if t == SAFE_HAVEN_TICKER:
            # SHV: nearly flat with tiny positive drift
            returns = np.random.normal(0.0001, 0.001, n_days)
        elif t == "SPY":
            # SPY: moderate uptrend
            returns = np.random.normal(0.0004, 0.012, n_days)
        else:
            # Regular tickers: varying drift and volatility
            drift = 0.0003 * (1 + (i % 5) * 0.3)
            vol = 0.015 * (1 + (i % 4) * 0.25)
            returns = np.random.normal(drift, vol, n_days)

        prices = 100 * np.exp(np.cumsum(returns))
        data[t] = prices

    return pd.DataFrame(data, index=dates)


@pytest.fixture
def synthetic_open_prices(synthetic_prices: pd.DataFrame) -> pd.DataFrame:
    """Open prices: Close shifted slightly with noise."""
    np.random.seed(123)
    noise = 1 + np.random.normal(0, 0.002, synthetic_prices.shape)
    return synthetic_prices * noise


@pytest.fixture
def tickers_list(synthetic_prices: pd.DataFrame) -> list[str]:
    """Ordered list of tickers from synthetic data."""
    return list(synthetic_prices.columns)
