"""Shared test fixtures for portfolio simulation tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    """Generate 500 days of synthetic Close prices for 20 tickers + SPY."""
    np.random.seed(42)
    n_days = 500
    tickers = [f"T{i:02d}" for i in range(20)] + ["SPY"]

    dates = pd.bdate_range("2021-01-04", periods=n_days)
    data = {}

    for i, t in enumerate(tickers):
        if t == "SPY":
            returns = np.random.normal(0.0004, 0.012, n_days)
        else:
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
    """Tradable tickers (excludes SPY)."""
    return [t for t in synthetic_prices.columns if t != "SPY"]
