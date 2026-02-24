"""Tests for simplified simulation engine (Long/Cash only)."""

import numpy as np
import pandas as pd
import pytest

from src.portfolio_sim.config import INITIAL_CAPITAL
from src.portfolio_sim.engine import COST_RATE, _execute_trades, run_simulation


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


def test_execute_trades_bear_sellall_no_double_counting():
    """Bear regime sell-all must return equity_at_open minus costs, not more."""
    shares = {"AAPL": 10.0, "MSFT": 5.0}
    open_prices = pd.Series({"AAPL": 150.0, "MSFT": 300.0})

    # equity_at_open = outer_cash(2000) + 10*150 + 5*300 = 5000
    equity_at_open = 5000.0
    trades = {}  # empty dict = sell everything (bear regime)

    returned_cash = _execute_trades(shares, trades, equity_at_open, open_prices)

    # total trade value = 10*150 + 5*300 = 3000
    expected_cost = 3000.0 * COST_RATE
    expected_cash = equity_at_open - expected_cost

    assert shares == {}
    assert returned_cash == pytest.approx(expected_cash)
    assert returned_cash < equity_at_open


def test_execute_trades_bear_sellall_no_positions():
    """Bear sell-all with no positions should return equity minus zero costs."""
    shares = {}
    open_prices = pd.Series(dtype=float)
    equity_at_open = 10000.0
    trades = {}

    returned_cash = _execute_trades(shares, trades, equity_at_open, open_prices)
    assert returned_cash == pytest.approx(equity_at_open)


def test_bear_regime_equity_does_not_inflate():
    """Equity must never significantly exceed initial capital in a pure bear scenario."""
    np.random.seed(99)
    n = 200
    dates = pd.bdate_range("2022-01-03", periods=n)
    tickers = [f"T{i:02d}" for i in range(5)]

    data = {}
    for t in tickers:
        data[t] = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.005, n)))
    # SPY in massive steady decline — triggers bear regime
    data["SPY"] = 100 * np.exp(np.cumsum(np.full(n, -0.01)))

    close = pd.DataFrame(data, index=dates)
    open_ = close.copy()

    equity, _ = run_simulation(close, open_, tickers, INITIAL_CAPITAL)

    # With bear sell-all and costs, equity should not inflate beyond initial capital
    assert equity.max() < INITIAL_CAPITAL * 1.05
