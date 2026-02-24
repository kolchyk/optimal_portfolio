"""Tests for simulation engine (Long/Cash only) with lazy hold + hysteresis."""

import numpy as np
import pandas as pd
import pytest

from src.portfolio_sim.config import INITIAL_CAPITAL, TOP_N
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
    result = run_simulation(close, open_, tickers, INITIAL_CAPITAL)
    assert isinstance(result.equity, pd.Series)
    assert isinstance(result.spy_equity, pd.Series)
    assert len(result.equity) == len(result.spy_equity)


def test_equity_starts_near_initial_capital(sim_data):
    close, open_, tickers = sim_data
    result = run_simulation(close, open_, tickers, INITIAL_CAPITAL)
    equity = result.equity
    assert abs(equity.iloc[0] - INITIAL_CAPITAL) / INITIAL_CAPITAL < 0.05


def test_spy_benchmark_starts_at_initial_capital(sim_data):
    close, open_, tickers = sim_data
    result = run_simulation(close, open_, tickers, INITIAL_CAPITAL)
    spy_eq = result.spy_equity
    assert spy_eq.iloc[0] == pytest.approx(INITIAL_CAPITAL, rel=0.01)


def test_equity_positive(sim_data):
    close, open_, tickers = sim_data
    result = run_simulation(close, open_, tickers, INITIAL_CAPITAL)
    equity = result.equity
    assert (equity >= 0).all()


def test_bear_regime_preserves_capital():
    """When SPY is in strong downtrend, strategy should go to cash."""
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

    result = run_simulation(close, open_, tickers, INITIAL_CAPITAL)
    strat_return = result.equity.iloc[-1] / result.equity.iloc[0] - 1
    spy_return = result.spy_equity.iloc[-1] / result.spy_equity.iloc[0] - 1
    assert strat_return > spy_return


def test_execute_trades_bear_sellall_no_double_counting():
    """Bear regime sell-all must return equity_at_open minus costs, not more."""
    shares = {"AAPL": 10.0, "MSFT": 5.0}
    open_prices = pd.Series({"AAPL": 150.0, "MSFT": 300.0})

    equity_at_open = 5000.0
    trades = {}

    returned_cash = _execute_trades(shares, trades, equity_at_open, open_prices)

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
    data["SPY"] = 100 * np.exp(np.cumsum(np.full(n, -0.01)))

    close = pd.DataFrame(data, index=dates)
    open_ = close.copy()

    result = run_simulation(close, open_, tickers, INITIAL_CAPITAL)
    equity = result.equity
    assert equity.max() < INITIAL_CAPITAL * 1.05


def test_strict_slot_sizing_single_buy():
    """FIX #3: A single buy must get at most 1/TOP_N of equity, not all cash."""
    shares = {}
    open_prices = pd.Series({"AAPL": 100.0})
    equity_at_open = 10000.0
    trades = {"AAPL": 1.0}

    _execute_trades(shares, trades, equity_at_open, open_prices)

    max_slot = equity_at_open / TOP_N
    position_value = shares["AAPL"] * 100.0
    assert position_value <= max_slot * 1.01


def test_strict_slot_sizing_prevents_concentration():
    """With freed cash and 1 buy, position must still be <= 1/TOP_N of equity."""
    shares = {"MSFT": 20.0}
    open_prices = pd.Series({"MSFT": 250.0, "GOOG": 100.0})
    equity_at_open = 10000.0
    trades = {"GOOG": 1.0}

    _execute_trades(shares, trades, equity_at_open, open_prices)

    max_slot = equity_at_open / TOP_N
    goog_value = shares["GOOG"] * 100.0
    assert goog_value <= max_slot * 1.01
