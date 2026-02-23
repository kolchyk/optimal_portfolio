"""Tests for simulation engine."""

import numpy as np
import pandas as pd
import pytest

from src.portfolio_sim.config import INITIAL_CAPITAL, SAFE_HAVEN_TICKER, StrategyParams
from src.portfolio_sim.engine import run_simulation


@pytest.fixture
def short_sim_data():
    """Create minimal simulation data: 50 sim days with 200 days full history."""
    np.random.seed(42)
    n_full = 250
    n_sim = 50
    dates = pd.bdate_range("2022-01-03", periods=n_full)
    tickers = [f"T{i:02d}" for i in range(10)] + [SAFE_HAVEN_TICKER]

    data = {}
    for t in tickers:
        if t == SAFE_HAVEN_TICKER:
            data[t] = np.linspace(100, 100.5, n_full)
        else:
            data[t] = 100 * np.exp(
                np.cumsum(np.random.normal(0.0003, 0.015, n_full))
            )

    full_close = pd.DataFrame(data, index=dates)
    full_open = full_close * (1 + np.random.normal(0, 0.001, full_close.shape))

    sim_close = full_close.iloc[-n_sim:]
    sim_open = full_open.iloc[-n_sim:]

    return sim_close, sim_open, full_close, tickers


def test_equity_length(short_sim_data):
    sim_close, sim_open, full_close, tickers = short_sim_data
    params = StrategyParams(lookback_period=63)
    equity, exposures, weights = run_simulation(
        sim_close, sim_open, full_close, tickers, params, INITIAL_CAPITAL
    )
    assert len(equity) == len(sim_close)
    assert len(exposures) == len(sim_close)


def test_equity_starts_near_initial_capital(short_sim_data):
    sim_close, sim_open, full_close, tickers = short_sim_data
    params = StrategyParams(lookback_period=63)
    equity, _, _ = run_simulation(
        sim_close, sim_open, full_close, tickers, params, INITIAL_CAPITAL
    )
    # First day equity should be close to initial capital (minus small commission)
    assert abs(equity[0] - INITIAL_CAPITAL) / INITIAL_CAPITAL < 0.05


def test_exposure_bounded(short_sim_data):
    sim_close, sim_open, full_close, tickers = short_sim_data
    params = StrategyParams(lookback_period=63)
    _, exposures, _ = run_simulation(
        sim_close, sim_open, full_close, tickers, params, INITIAL_CAPITAL
    )
    for exp in exposures:
        assert 0.0 <= exp <= 1.5  # Reasonable bounds


def test_weights_shape(short_sim_data):
    sim_close, sim_open, full_close, tickers = short_sim_data
    params = StrategyParams(lookback_period=63)
    _, _, weights = run_simulation(
        sim_close, sim_open, full_close, tickers, params, INITIAL_CAPITAL
    )
    assert len(weights) == len(tickers)


def test_bear_market_goes_to_shv():
    """When all prices are below KAMA, breadth < threshold → 100% SHV."""
    np.random.seed(42)
    n_full = 250
    n_sim = 30
    dates = pd.bdate_range("2022-01-03", periods=n_full)
    tickers = [f"T{i:02d}" for i in range(10)] + [SAFE_HAVEN_TICKER]

    data = {}
    for t in tickers:
        if t == SAFE_HAVEN_TICKER:
            data[t] = np.linspace(100, 100.5, n_full)
        else:
            # Strong downtrend: price will be well below KAMA
            data[t] = 100 * np.exp(
                np.cumsum(np.random.normal(-0.005, 0.01, n_full))
            )

    full_close = pd.DataFrame(data, index=dates)
    full_open = full_close.copy()

    sim_close = full_close.iloc[-n_sim:]
    sim_open = full_open.iloc[-n_sim:]

    params = StrategyParams(lookback_period=63, kama_period=10)
    equity, exposures, _ = run_simulation(
        sim_close, sim_open, full_close, tickers, params, INITIAL_CAPITAL
    )

    # In a severe bear market, long exposure should be very low (mostly SHV)
    avg_exposure = np.mean(exposures[-10:])
    assert avg_exposure < 0.5
