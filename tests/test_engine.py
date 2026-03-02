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
    high = close * (1 + np.abs(np.random.normal(0, 0.005, close.shape)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, close.shape)))
    return close, open_, tickers, high, low


# ---------------------------------------------------------------------------
# run_simulation integration tests
# ---------------------------------------------------------------------------
def test_returns_simulation_result(sim_data):
    close, open_, tickers, high, low = sim_data
    result = run_simulation(close, open_, tickers, INITIAL_CAPITAL,
                            high_prices=high, low_prices=low)
    assert isinstance(result, SimulationResult)
    assert isinstance(result.equity, pd.Series)
    assert isinstance(result.spy_equity, pd.Series)
    assert len(result.equity) == len(result.spy_equity)


def test_equity_starts_near_initial_capital(sim_data):
    close, open_, tickers, high, low = sim_data
    result = run_simulation(close, open_, tickers, INITIAL_CAPITAL,
                            high_prices=high, low_prices=low)
    assert abs(result.equity.iloc[0] - INITIAL_CAPITAL) / INITIAL_CAPITAL < 0.10


def test_spy_benchmark_starts_at_initial_capital(sim_data):
    close, open_, tickers, high, low = sim_data
    result = run_simulation(close, open_, tickers, INITIAL_CAPITAL,
                            high_prices=high, low_prices=low)
    assert result.spy_equity.iloc[0] == pytest.approx(INITIAL_CAPITAL, rel=0.01)


def test_equity_positive(sim_data):
    close, open_, tickers, high, low = sim_data
    result = run_simulation(close, open_, tickers, INITIAL_CAPITAL,
                            high_prices=high, low_prices=low)
    assert (result.equity >= 0).all()


def test_holdings_history_is_dataframe(sim_data):
    close, open_, tickers, high, low = sim_data
    result = run_simulation(close, open_, tickers, INITIAL_CAPITAL,
                            high_prices=high, low_prices=low)
    assert isinstance(result.holdings_history, pd.DataFrame)


def test_cash_history_is_series(sim_data):
    close, open_, tickers, high, low = sim_data
    result = run_simulation(close, open_, tickers, INITIAL_CAPITAL,
                            high_prices=high, low_prices=low)
    assert isinstance(result.cash_history, pd.Series)


def test_trade_log_is_list(sim_data):
    close, open_, tickers, high, low = sim_data
    result = run_simulation(close, open_, tickers, INITIAL_CAPITAL,
                            high_prices=high, low_prices=low)
    assert isinstance(result.trade_log, list)


def test_custom_params(sim_data):
    close, open_, tickers, high, low = sim_data
    params = StrategyParams(top_n=3, rebal_days=10, target_vol=0.15)
    result = run_simulation(close, open_, tickers, INITIAL_CAPITAL, params=params,
                            high_prices=high, low_prices=low)
    assert isinstance(result, SimulationResult)
    assert not result.equity.empty


def test_fallback_without_high_low(sim_data):
    """Without high/low, should fall back to close-to-close ATR."""
    close, open_, tickers, _high, _low = sim_data
    result = run_simulation(close, open_, tickers, INITIAL_CAPITAL)
    assert isinstance(result, SimulationResult)
    assert not result.equity.empty


def test_cost_rate_positive():
    assert COST_RATE > 0


def test_max_per_class_limits_concentration(sim_data):
    """Per-class limit prevents loading >max_per_class from one class."""
    close, open_, tickers, high, low = sim_data

    import src.portfolio_sim.config as cfg
    original_map = cfg.ASSET_CLASS_MAP.copy()
    for t in tickers:
        cfg.ASSET_CLASS_MAP[t] = "TestClass"

    try:
        params = StrategyParams(
            top_n=10,
            max_per_class=2,
        )
        result = run_simulation(
            close, open_, tickers, INITIAL_CAPITAL,
            params=params,
            high_prices=high, low_prices=low,
        )
        # At no point should more than 2 positions be held
        active_counts = (result.holdings_history > 0).sum(axis=1)
        assert active_counts.max() <= 2
    finally:
        cfg.ASSET_CLASS_MAP.clear()
        cfg.ASSET_CLASS_MAP.update(original_map)


def test_max_per_class_allows_multiple_classes(sim_data):
    """With 2 classes and max_per_class=2, can hold up to 4 positions."""
    close, open_, tickers, high, low = sim_data

    import src.portfolio_sim.config as cfg
    original_map = cfg.ASSET_CLASS_MAP.copy()
    for i, t in enumerate(tickers):
        cfg.ASSET_CLASS_MAP[t] = "ClassA" if i < 5 else "ClassB"

    try:
        params = StrategyParams(
            top_n=10,
            max_per_class=2,
        )
        result = run_simulation(
            close, open_, tickers, INITIAL_CAPITAL,
            params=params,
            high_prices=high, low_prices=low,
        )
        active_counts = (result.holdings_history > 0).sum(axis=1)
        # Can hold up to 4 (2 from ClassA + 2 from ClassB), but not 10
        assert active_counts.max() <= 4
    finally:
        cfg.ASSET_CLASS_MAP.clear()
        cfg.ASSET_CLASS_MAP.update(original_map)


# ---------------------------------------------------------------------------
# min_invested_pct tests
# ---------------------------------------------------------------------------
def test_min_invested_pct_default_unchanged(sim_data):
    """Default min_invested_pct=0.0 produces identical results to no floor."""
    close, open_, tickers, high, low = sim_data
    params_default = StrategyParams()
    params_zero = StrategyParams(min_invested_pct=0.0)

    r1 = run_simulation(close, open_, tickers, INITIAL_CAPITAL,
                         params=params_default, high_prices=high, low_prices=low)
    r2 = run_simulation(close, open_, tickers, INITIAL_CAPITAL,
                         params=params_zero, high_prices=high, low_prices=low)

    pd.testing.assert_series_equal(r1.equity, r2.equity)


def test_min_invested_pct_increases_investment(sim_data):
    """Setting min_invested_pct=0.8 should result in less cash on average."""
    close, open_, tickers, high, low = sim_data

    r_default = run_simulation(
        close, open_, tickers, INITIAL_CAPITAL,
        params=StrategyParams(),
        high_prices=high, low_prices=low,
    )
    r_floor = run_simulation(
        close, open_, tickers, INITIAL_CAPITAL,
        params=StrategyParams(min_invested_pct=0.8),
        high_prices=high, low_prices=low,
    )

    # Average cash ratio should be lower with the floor
    avg_cash_default = (r_default.cash_history / r_default.equity).mean()
    avg_cash_floor = (r_floor.cash_history / r_floor.equity).mean()
    assert avg_cash_floor < avg_cash_default


def test_min_invested_pct_validation():
    """min_invested_pct outside [0, 1] should raise ValueError."""
    with pytest.raises(ValueError, match="min_invested_pct"):
        StrategyParams(min_invested_pct=1.5)
    with pytest.raises(ValueError, match="min_invested_pct"):
        StrategyParams(min_invested_pct=-0.1)
