"""Tests for hybrid simulation engine."""

import numpy as np
import pandas as pd
import pytest

from src.portfolio_sim.config import INITIAL_CAPITAL
from src.portfolio_sim.engine import COST_RATE, run_simulation
from src.portfolio_sim.models import SimulationResult
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


# ---------------------------------------------------------------------------
# run_simulation integration tests
# ---------------------------------------------------------------------------
def test_returns_simulation_result(sim_data):
    close, open_, tickers = sim_data
    result = run_simulation(close, open_, tickers, INITIAL_CAPITAL)
    assert isinstance(result, SimulationResult)
    assert isinstance(result.equity, pd.Series)
    assert isinstance(result.spy_equity, pd.Series)
    assert len(result.equity) == len(result.spy_equity)


def test_equity_starts_near_initial_capital(sim_data):
    close, open_, tickers = sim_data
    result = run_simulation(close, open_, tickers, INITIAL_CAPITAL)
    assert abs(result.equity.iloc[0] - INITIAL_CAPITAL) / INITIAL_CAPITAL < 0.10


def test_spy_benchmark_starts_at_initial_capital(sim_data):
    close, open_, tickers = sim_data
    result = run_simulation(close, open_, tickers, INITIAL_CAPITAL)
    assert result.spy_equity.iloc[0] == pytest.approx(INITIAL_CAPITAL, rel=0.01)


def test_equity_positive(sim_data):
    close, open_, tickers = sim_data
    result = run_simulation(close, open_, tickers, INITIAL_CAPITAL)
    assert (result.equity >= 0).all()


def test_holdings_history_is_dataframe(sim_data):
    close, open_, tickers = sim_data
    result = run_simulation(close, open_, tickers, INITIAL_CAPITAL)
    assert isinstance(result.holdings_history, pd.DataFrame)


def test_cash_history_is_series(sim_data):
    close, open_, tickers = sim_data
    result = run_simulation(close, open_, tickers, INITIAL_CAPITAL)
    assert isinstance(result.cash_history, pd.Series)


def test_trade_log_is_list(sim_data):
    close, open_, tickers = sim_data
    result = run_simulation(close, open_, tickers, INITIAL_CAPITAL)
    assert isinstance(result.trade_log, list)


def test_custom_params(sim_data):
    close, open_, tickers = sim_data
    params = StrategyParams(top_n=3, rebal_period_weeks=2, target_vol=0.15)
    result = run_simulation(close, open_, tickers, INITIAL_CAPITAL, params=params)
    assert isinstance(result, SimulationResult)
    assert not result.equity.empty


def test_cost_rate_positive():
    assert COST_RATE > 0
