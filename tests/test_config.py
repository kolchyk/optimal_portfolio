"""Tests for config module."""

from src.portfolio_sim.config import (
    ETF_UNIVERSE,
    INITIAL_CAPITAL,
    KAMA_BUFFER,
    KAMA_PERIOD,
    R2_LOOKBACK,
    SPY_TICKER,
    TOP_N,
)


def test_constants_values():
    assert INITIAL_CAPITAL == 10_000
    assert KAMA_PERIOD == 40
    assert R2_LOOKBACK == 90
    assert TOP_N == 10
    assert KAMA_BUFFER == 0.01
    assert SPY_TICKER == "SPY"


def test_etf_universe_lean():
    # Crypto ETFs present
    assert "IBIT" in ETF_UNIVERSE
    assert "ETHA" in ETF_UNIVERSE
    # Old raw crypto tickers removed
    assert "BTC-USD" not in ETF_UNIVERSE
    assert "ETH-USD" not in ETF_UNIVERSE
    # Lean universe ~37 tickers
    assert len(ETF_UNIVERSE) == 37


def test_constants_types():
    assert isinstance(KAMA_PERIOD, int)
    assert isinstance(R2_LOOKBACK, int)
    assert isinstance(TOP_N, int)
    assert isinstance(KAMA_BUFFER, float)
