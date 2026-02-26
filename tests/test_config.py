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
    assert TOP_N == 10
    assert KAMA_BUFFER == 0.024
    assert SPY_TICKER == "SPY"


def test_etf_universe_crypto():
    from src.portfolio_sim.config import ETF_UNIVERSE
    crypto_etfs = ["IBIT", "FBTC", "GBTC", "ARKB", "ETHA"]
    # After implementation, these should be in the universe
    for etf in crypto_etfs:
        assert etf in ETF_UNIVERSE
    
    # And old tickers should NOT be in the universe
    assert "BTC-USD" not in ETF_UNIVERSE
    assert "ETH-USD" not in ETF_UNIVERSE


def test_constants_types():
    assert isinstance(KAMA_PERIOD, int)
    assert isinstance(LOOKBACK_PERIOD, int)
    assert isinstance(TOP_N, int)
    assert isinstance(KAMA_BUFFER, float)
