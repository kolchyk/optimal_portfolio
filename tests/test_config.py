"""Tests for config module."""

from src.portfolio_sim.config import (
    INITIAL_CAPITAL,
    KAMA_BUFFER,
    KAMA_PERIOD,
    LOOKBACK_PERIOD,
    SPY_TICKER,
    TOP_N,
)


def test_constants_values():
    assert INITIAL_CAPITAL == 10_000
    assert KAMA_PERIOD == 20
    assert LOOKBACK_PERIOD == 60
    assert TOP_N == 20
    assert KAMA_BUFFER == 0.01
    assert SPY_TICKER == "SPY"


def test_constants_types():
    assert isinstance(KAMA_PERIOD, int)
    assert isinstance(LOOKBACK_PERIOD, int)
    assert isinstance(TOP_N, int)
    assert isinstance(KAMA_BUFFER, float)
