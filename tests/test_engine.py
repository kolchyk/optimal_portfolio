"""Tests for simulation engine (Long/Cash only) with lazy hold + hysteresis."""

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


def test_bear_regime_preserves_capital():
    """When SPY is in strong downtrend AND regime filter is ON, strategy should go to cash."""
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

    params = StrategyParams(enable_regime_filter=True)
    result = run_simulation(close, open_, tickers, INITIAL_CAPITAL, params=params)
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
    """Equity must never significantly exceed initial capital in a pure bear scenario
    when regime filter is enabled."""
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

    params = StrategyParams(enable_regime_filter=True)
    result = run_simulation(close, open_, tickers, INITIAL_CAPITAL, params=params)
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


# ---------------------------------------------------------------------------
# New tests: regime filter disabled
# ---------------------------------------------------------------------------
def test_no_regime_filter_holds_during_spy_crash():
    """Without regime filter, individual positions stay even when SPY crashes."""
    np.random.seed(42)
    n = 250
    dates = pd.bdate_range("2022-01-03", periods=n)
    tickers = [f"T{i:02d}" for i in range(5)]

    data = {}
    # Tickers in strong uptrend
    for t in tickers:
        data[t] = 100 * np.exp(
            np.cumsum(np.random.normal(0.003, 0.01, n))
        )
    # SPY crashes
    data["SPY"] = 100 * np.exp(np.cumsum(np.full(n, -0.005)))

    close = pd.DataFrame(data, index=dates)
    open_ = close.copy()

    params = StrategyParams(enable_regime_filter=False)
    result = run_simulation(close, open_, tickers, INITIAL_CAPITAL, params=params)

    # Should still have holdings (not all liquidated)
    final_holdings = result.holdings_history.iloc[-1]
    assert final_holdings.sum() > 0, "Positions should be held when regime filter is off"


def test_regime_history_none_when_filter_disabled(sim_data):
    """regime_history should be None when enable_regime_filter=False."""
    close, open_, tickers = sim_data
    params = StrategyParams(enable_regime_filter=False)
    result = run_simulation(close, open_, tickers, INITIAL_CAPITAL, params=params)
    assert result.regime_history is None


def test_regime_history_present_when_filter_enabled(sim_data):
    """regime_history should be a Series when enable_regime_filter=True."""
    close, open_, tickers = sim_data
    params = StrategyParams(enable_regime_filter=True)
    result = run_simulation(close, open_, tickers, INITIAL_CAPITAL, params=params)
    assert result.regime_history is not None
    assert isinstance(result.regime_history, pd.Series)


# ---------------------------------------------------------------------------
# New tests: risk parity sizing
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
