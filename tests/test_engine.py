"""Tests for simplified simulation engine (Long/Cash only)."""

import numpy as np
import pandas as pd
import pytest

from src.portfolio_sim.config import INITIAL_CAPITAL
from src.portfolio_sim.engine import run_simulation


@pytest.fixture
def sim_data():
    """250 days of full history for 10 tickers + SPY."""
    np.random.seed(42)
    n_full = 250
    dates = pd.bdate_range("2022-01-03", periods=n_full)
    tickers = [f"T{i:02d}" for i in range(10)]

    data = {}
    for t in tickers:
        data[t] = 100 * np.exp(
            np.cumsum(np.random.normal(0.0003, 0.015, n_full))
        )
    data["SPY"] = 100 * np.exp(
        np.cumsum(np.random.normal(0.0004, 0.01, n_full))
    )

    close = pd.DataFrame(data, index=dates)
    open_ = close * (1 + np.random.normal(0, 0.001, close.shape))
    return close, open_, tickers


def test_returns_two_series(sim_data):
    close, open_, tickers = sim_data
    equity, spy_eq = run_simulation(close, open_, tickers, INITIAL_CAPITAL)
    assert isinstance(equity, pd.Series)
    assert isinstance(spy_eq, pd.Series)
    assert len(equity) == len(spy_eq)


def test_equity_starts_near_initial_capital(sim_data):
    close, open_, tickers = sim_data
    equity, _ = run_simulation(close, open_, tickers, INITIAL_CAPITAL)
    assert abs(equity.iloc[0] - INITIAL_CAPITAL) / INITIAL_CAPITAL < 0.05


def test_spy_benchmark_starts_at_initial_capital(sim_data):
    close, open_, tickers = sim_data
    _, spy_eq = run_simulation(close, open_, tickers, INITIAL_CAPITAL)
    assert spy_eq.iloc[0] == pytest.approx(INITIAL_CAPITAL, rel=0.01)


def test_equity_positive(sim_data):
    close, open_, tickers = sim_data
    equity, _ = run_simulation(close, open_, tickers, INITIAL_CAPITAL)
    assert (equity >= 0).all()


def test_bear_regime_preserves_capital():
    """When SPY is in strong downtrend, strategy should go to cash and preserve capital."""
    np.random.seed(42)
    n = 250
    dates = pd.bdate_range("2022-01-03", periods=n)
    tickers = [f"T{i:02d}" for i in range(5)]

    data = {}
    for t in tickers:
        data[t] = 100 * np.exp(
            np.cumsum(np.random.normal(-0.005, 0.01, n))
        )
    data["SPY"] = 100 * np.exp(
        np.cumsum(np.random.normal(-0.005, 0.01, n))
    )

    close = pd.DataFrame(data, index=dates)
    open_ = close.copy()

    equity, spy_eq = run_simulation(close, open_, tickers, INITIAL_CAPITAL)
    # In pure bear market, strategy should lose less than SPY by going to cash
    strat_return = equity.iloc[-1] / equity.iloc[0] - 1
    spy_return = spy_eq.iloc[-1] / spy_eq.iloc[0] - 1
    assert strat_return > spy_return
