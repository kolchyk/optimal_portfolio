"""Tests for config module."""

from src.portfolio_sim.config import (
    ETF_UNIVERSE,
    INITIAL_CAPITAL,
    PARAM_NAMES,
    SEARCH_SPACE,
    SPY_TICKER,
)


def test_constants_values():
    assert INITIAL_CAPITAL == 10_000
    assert SPY_TICKER == "SPY"


def test_search_space_has_expected_keys():
    assert "r2_lookback" in SEARCH_SPACE
    assert "target_vol" in SEARCH_SPACE
    assert "kama_asset_period" in SEARCH_SPACE
    assert "corr_threshold" in SEARCH_SPACE


def test_param_names_match_search_space():
    assert set(PARAM_NAMES) == set(SEARCH_SPACE.keys())


def test_etf_universe_lean():
    # Crypto ETFs present
    assert "IBIT" in ETF_UNIVERSE
    assert "ETHA" in ETF_UNIVERSE
    # Old raw crypto tickers removed
    assert "BTC-USD" not in ETF_UNIVERSE
    assert "ETH-USD" not in ETF_UNIVERSE
    # Lean universe ~37 tickers
    assert len(ETF_UNIVERSE) == 37
