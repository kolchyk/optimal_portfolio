"""Tests for simulation engine (Long/Cash only) with lazy hold."""

import numpy as np
import pandas as pd
import pytest

from src.portfolio_sim.config import INITIAL_CAPITAL, TOP_N
from src.portfolio_sim.engine import COST_RATE, _execute_trades, run_simulation
from src.portfolio_sim.params import StrategyParams


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


# ---------------------------------------------------------------------------
# Risk parity sizing
# ---------------------------------------------------------------------------
def test_execute_trades_with_weights():
    """Verify _execute_trades respects provided risk parity weights."""
    shares = {}
    open_prices = pd.Series({"A": 100.0, "B": 100.0})
    equity_at_open = 10000.0
    trades = {"A": 1.0, "B": 1.0}
    # A gets 70%, B gets 30% (simulating inverse vol weighting)
    weights = {"A": 0.7, "B": 0.3}

    _execute_trades(shares, trades, equity_at_open, open_prices, top_n=2, weights=weights)

    value_a = shares["A"] * 100.0
    value_b = shares["B"] * 100.0
    assert value_a > value_b, "Higher weight should get larger allocation"
    assert value_a == pytest.approx(equity_at_open * 0.7 / (1 + COST_RATE), rel=0.01)
    assert value_b == pytest.approx(equity_at_open * 0.3 / (1 + COST_RATE), rel=0.01)


def test_execute_trades_without_weights_uses_equal():
    """Without weights, each buy gets 1/top_n."""
    shares = {}
    open_prices = pd.Series({"A": 100.0, "B": 100.0})
    equity_at_open = 10000.0
    trades = {"A": 1.0, "B": 1.0}

    _execute_trades(shares, trades, equity_at_open, open_prices, top_n=2, weights=None)

    value_a = shares["A"] * 100.0
    value_b = shares["B"] * 100.0
    max_per_slot = equity_at_open / 2
    assert value_a == pytest.approx(max_per_slot / (1 + COST_RATE), rel=0.01)
    assert value_b == pytest.approx(max_per_slot / (1 + COST_RATE), rel=0.01)
