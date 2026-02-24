"""Tests for simplified KAMA momentum alpha (Long/Cash, equal weight)."""

import numpy as np
import pandas as pd
import pytest

from src.portfolio_sim.alpha import compute_target_weights
from src.portfolio_sim.config import TOP_N


@pytest.fixture
def simple_prices():
    """10 tickers, 200 days, with clear momentum differences."""
    np.random.seed(42)
    n_days = 200
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    data = {}
    # T00-T04: strong uptrend (good momentum)
    for i in range(5):
        drift = 0.001 * (5 - i)
        data[f"T{i:02d}"] = 100 * np.exp(
            np.cumsum(np.random.normal(drift, 0.01, n_days))
        )
    # T05-T09: downtrend (bad momentum)
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


def test_returns_dict(simple_prices, simple_tickers, kama_low):
    result = compute_target_weights(simple_prices, simple_tickers, kama_low)
    assert isinstance(result, dict)


def test_bull_mode_equal_weight(simple_prices, simple_tickers, kama_low):
    result = compute_target_weights(simple_prices, simple_tickers, kama_low)
    if result:
        expected_weight = 1.0 / TOP_N
        for w in result.values():
            assert w == pytest.approx(expected_weight)


def test_bear_mode_returns_empty(simple_prices, simple_tickers, kama_low):
    result = compute_target_weights(
        simple_prices, simple_tickers, kama_low, is_bull=False
    )
    assert result == {}


def test_kama_filter_removes_all(simple_prices, simple_tickers, kama_high):
    result = compute_target_weights(simple_prices, simple_tickers, kama_high)
    assert result == {}


def test_positive_momentum_preferred(simple_prices, simple_tickers, kama_low):
    result = compute_target_weights(simple_prices, simple_tickers, kama_low)
    # T00-T04 have strong positive drift, they should dominate the selection
    strong_uptrend = {f"T{i:02d}" for i in range(3)}  # T00, T01, T02
    selected = set(result.keys())
    assert strong_uptrend.issubset(selected)


def test_max_positions_capped(simple_prices, simple_tickers, kama_low):
    result = compute_target_weights(simple_prices, simple_tickers, kama_low)
    assert len(result) <= TOP_N


def test_weights_sum_le_one(simple_prices, simple_tickers, kama_low):
    result = compute_target_weights(simple_prices, simple_tickers, kama_low)
    assert sum(result.values()) <= 1.0 + 1e-6
