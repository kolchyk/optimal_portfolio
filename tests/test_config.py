"""Tests for config module."""

import pytest

from src.portfolio_sim.config import (
    BREADTH_THRESHOLD,
    INITIAL_CAPITAL,
    MAX_WEIGHT,
    REBALANCE_INTERVAL,
    VOL_FLOOR,
    StrategyParams,
)


def test_strategy_params_defaults():
    params = StrategyParams()
    assert params.kama_period == 20
    assert params.lookback_period == 126
    assert params.max_correlation == 0.70
    assert params.top_n_selection == 15


def test_strategy_params_custom():
    params = StrategyParams(kama_period=30, top_n_selection=25)
    assert params.kama_period == 30
    assert params.top_n_selection == 25
    # Others remain default
    assert params.lookback_period == 126


def test_strategy_params_frozen():
    params = StrategyParams()
    with pytest.raises(AttributeError):
        params.kama_period = 30  # type: ignore[misc]


def test_constants():
    assert INITIAL_CAPITAL == 10_000
    assert MAX_WEIGHT == 0.15
    assert BREADTH_THRESHOLD == 0.30
    assert VOL_FLOOR == 0.05
    assert REBALANCE_INTERVAL == 21
