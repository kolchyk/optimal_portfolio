"""Tests for KAMA momentum alpha — get_buy_candidates API."""

import numpy as np
import pandas as pd
import pytest

from src.portfolio_sim.alpha import get_buy_candidates
from src.portfolio_sim.config import TOP_N


@pytest.fixture
def simple_prices():
    """10 tickers, 200 days, with clear momentum differences."""
    np.random.seed(42)
    n_days = 200
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    data = {}
    for i in range(5):
        drift = 0.001 * (5 - i)
        data[f"T{i:02d}"] = 100 * np.exp(
            np.cumsum(np.random.normal(drift, 0.01, n_days))
        )
    for i in range(5, 10):
        data[f"T{i:02d}"] = 100 * np.exp(
            np.cumsum(np.random.normal(-0.001, 0.01, n_days))
        )
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def simple_tickers(simple_prices):
    return list(simple_prices.columns)


@pytest.fixture
def kama_low(simple_tickers):
    """KAMA set low so all tickers pass Close > KAMA * (1 + buffer)."""
    return {t: 0.0 for t in simple_tickers}


@pytest.fixture
def kama_high(simple_tickers):
    """KAMA set very high so all tickers FAIL the filter."""
    return {t: 1e6 for t in simple_tickers}


def test_returns_list(simple_prices, simple_tickers, kama_low):
    result = get_buy_candidates(simple_prices, simple_tickers, kama_low)
    assert isinstance(result, list)


def test_candidates_are_strings(simple_prices, simple_tickers, kama_low):
    result = get_buy_candidates(simple_prices, simple_tickers, kama_low)
    assert all(isinstance(t, str) for t in result)


def test_kama_filter_removes_all(simple_prices, simple_tickers, kama_high):
    result = get_buy_candidates(simple_prices, simple_tickers, kama_high)
    assert result == []


def test_positive_momentum_preferred(simple_prices, simple_tickers, kama_low):
    result = get_buy_candidates(simple_prices, simple_tickers, kama_low)
    strong_uptrend = {f"T{i:02d}" for i in range(3)}
    assert strong_uptrend.issubset(set(result))


def test_max_positions_capped(simple_prices, simple_tickers, kama_low):
    result = get_buy_candidates(simple_prices, simple_tickers, kama_low)
    assert len(result) <= TOP_N


def test_ranked_descending_by_momentum(simple_prices, simple_tickers, kama_low):
    """Candidates should be sorted with highest momentum first."""
    result = get_buy_candidates(simple_prices, simple_tickers, kama_low)
    if len(result) >= 2:
        for i in range(len(result) - 1):
            t_a, t_b = result[i], result[i + 1]
            mom_a = simple_prices[t_a].iloc[-1] / simple_prices[t_a].iloc[0] - 1
            mom_b = simple_prices[t_b].iloc[-1] / simple_prices[t_b].iloc[0] - 1
            assert mom_a >= mom_b


def test_empty_tickers_returns_empty(simple_prices, kama_low):
    result = get_buy_candidates(simple_prices, [], kama_low)
    assert result == []
