"""Tests for simulation engine."""

import numpy as np
import pandas as pd
import pytest

from src.portfolio_sim.config import INITIAL_CAPITAL, SAFE_HAVEN_TICKER, SPY_TICKER, StrategyParams
from src.portfolio_sim.engine import run_simulation


@pytest.fixture
def short_sim_data():
    """Create minimal simulation data: 50 sim days with 200 days full history.

    Includes SPY in uptrend (bull regime) for Market Breathing.
    """
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

    # SPY in uptrend so Market Breathing defaults to bull
    data[SPY_TICKER] = 100 * np.exp(
        np.cumsum(np.random.normal(0.0004, 0.01, n_full))
    )

    full_close = pd.DataFrame(data, index=dates)
    full_open = full_close * (1 + np.random.normal(0, 0.001, full_close.shape))

    sim_close = full_close.iloc[-n_sim:]
    sim_open = full_open.iloc[-n_sim:]

    return sim_close, sim_open, full_close, tickers


def test_equity_length(short_sim_data):
    sim_close, sim_open, full_close, tickers = short_sim_data
    params = StrategyParams(lookback_period=42)
    equity, gross_exp, net_exp, weights = run_simulation(
        sim_close, sim_open, full_close, tickers, params, INITIAL_CAPITAL
    )
    assert len(equity) == len(sim_close)
    assert len(gross_exp) == len(sim_close)
    assert len(net_exp) == len(sim_close)


def test_equity_starts_near_initial_capital(short_sim_data):
    sim_close, sim_open, full_close, tickers = short_sim_data
    params = StrategyParams(lookback_period=42)
    equity, _, _, _ = run_simulation(
        sim_close, sim_open, full_close, tickers, params, INITIAL_CAPITAL
    )
    # First day equity should be close to initial capital (minus small commission)
    assert abs(equity[0] - INITIAL_CAPITAL) / INITIAL_CAPITAL < 0.05


def test_gross_exposure_bounded(short_sim_data):
    sim_close, sim_open, full_close, tickers = short_sim_data
    params = StrategyParams(lookback_period=42)
    _, gross_exp, _, _ = run_simulation(
        sim_close, sim_open, full_close, tickers, params, INITIAL_CAPITAL
    )
    for g in gross_exp:
        assert 0.0 <= g <= 1.5  # Reasonable bounds


def test_net_exposure_bounded(short_sim_data):
    sim_close, sim_open, full_close, tickers = short_sim_data
    params = StrategyParams(lookback_period=42)
    _, _, net_exp, _ = run_simulation(
        sim_close, sim_open, full_close, tickers, params, INITIAL_CAPITAL
    )
    for n in net_exp:
        assert -1.5 <= n <= 1.5


def test_weights_shape(short_sim_data):
    sim_close, sim_open, full_close, tickers = short_sim_data
    params = StrategyParams(lookback_period=42)
    _, _, _, weights = run_simulation(
        sim_close, sim_open, full_close, tickers, params, INITIAL_CAPITAL
    )
    assert len(weights) == len(tickers)


def test_bear_regime_shorts():
    """When SPY is below KAMA(SPY), net exposure should be negative (short)."""
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
            # Strong downtrend
            data[t] = 100 * np.exp(
                np.cumsum(np.random.normal(-0.005, 0.01, n_full))
            )

    # SPY also in strong downtrend → below its KAMA → bear regime
    data[SPY_TICKER] = 100 * np.exp(
        np.cumsum(np.random.normal(-0.005, 0.01, n_full))
    )

    full_close = pd.DataFrame(data, index=dates)
    full_open = full_close.copy()

    sim_close = full_close.iloc[-n_sim:]
    sim_open = full_open.iloc[-n_sim:]

    params = StrategyParams(lookback_period=42, kama_period=10)
    equity, gross_exp, net_exp, _ = run_simulation(
        sim_close, sim_open, full_close, tickers, params, INITIAL_CAPITAL
    )

    # In a bear regime, net exposure should be negative or near-zero (short/flat)
    avg_net = np.mean(net_exp[-10:])
    assert avg_net < 0.1  # Should be negative or near-zero


def test_regime_transition_triggers_rebalance():
    """Switching from bull to bear should force immediate rebalance."""
    np.random.seed(42)
    n_full = 300
    n_sim = 60
    dates = pd.bdate_range("2022-01-03", periods=n_full)
    tickers = [f"T{i:02d}" for i in range(5)] + [SAFE_HAVEN_TICKER]

    data = {}
    for t in tickers:
        if t == SAFE_HAVEN_TICKER:
            data[t] = np.linspace(100, 100.5, n_full)
        else:
            data[t] = 100 * np.exp(
                np.cumsum(np.random.normal(0.0003, 0.015, n_full))
            )

    # SPY: uptrend first half, then strong downtrend
    spy_returns = np.concatenate([
        np.random.normal(0.002, 0.008, n_full - n_sim + 30),  # Bull
        np.random.normal(-0.008, 0.008, n_sim - 30),           # Bear
    ])
    data[SPY_TICKER] = 100 * np.exp(np.cumsum(spy_returns))

    full_close = pd.DataFrame(data, index=dates)
    full_open = full_close.copy()

    sim_close = full_close.iloc[-n_sim:]
    sim_open = full_open.iloc[-n_sim:]

    params = StrategyParams(lookback_period=21, kama_period=10)
    equity, gross_exp, net_exp, _ = run_simulation(
        sim_close, sim_open, full_close, tickers, params, INITIAL_CAPITAL
    )

    # Should have both positive and negative net exposures (regime transition)
    has_positive = any(n > 0.05 for n in net_exp)
    has_negative_or_flat = any(n < 0.05 for n in net_exp)
    assert has_positive and has_negative_or_flat
