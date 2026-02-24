"""Tests for config module."""

import pytest

from src.portfolio_sim.config import (
    INITIAL_CAPITAL,
    MAX_GROSS_EXPOSURE,
    MAX_WEIGHT,
    REBALANCE_INTERVAL,
    SPY_TICKER,
    VOL_FLOOR,
    StrategyParams,
)


def test_strategy_params_defaults():
    params = StrategyParams()
    assert params.kama_period == 10
    assert params.lookback_period == 21
    assert params.top_n_selection == 10


def test_strategy_params_custom():
    params = StrategyParams(kama_period=15, top_n_selection=20)
    assert params.kama_period == 15
    assert params.top_n_selection == 20
    # Others remain default
    assert params.lookback_period == 21


def test_strategy_params_frozen():
    params = StrategyParams()
    with pytest.raises(AttributeError):
        params.kama_period = 30  # type: ignore[misc]


def test_constants():
    assert INITIAL_CAPITAL == 10_000
    assert MAX_WEIGHT == 0.15
    assert MAX_GROSS_EXPOSURE == 1.0
    assert VOL_FLOOR == 0.05
    assert REBALANCE_INTERVAL == 5
    assert SPY_TICKER == "SPY"
