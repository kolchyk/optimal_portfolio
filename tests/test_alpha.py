"""Tests for KAMA momentum alpha — get_buy_candidates API."""

import numpy as np
import pandas as pd
import pytest

from src.portfolio_sim.alpha import _greedy_correlation_filter, get_buy_candidates
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


# ---------------------------------------------------------------------------
# Correlation filter tests
# ---------------------------------------------------------------------------
def test_correlation_filter_removes_correlated():
    """Highly correlated tickers should be filtered out by greedy algorithm."""
    np.random.seed(42)
    n_days = 100
    dates = pd.bdate_range("2023-01-02", periods=n_days)

    base = np.cumsum(np.random.normal(0, 0.01, n_days))
    data = {
        "A": 100 * np.exp(base),
        "B": 100 * np.exp(base + np.random.normal(0, 0.001, n_days)),  # ~perfectly correlated with A
        "C": 100 * np.exp(np.cumsum(np.random.normal(0, 0.02, n_days))),  # uncorrelated
    }
    prices = pd.DataFrame(data, index=dates)

    result = _greedy_correlation_filter(
        ranked_tickers=["A", "B", "C"],
        prices_window=prices,
        top_n=3,
        correlation_threshold=0.65,
        correlation_lookback=60,
    )
    assert "A" in result, "First ranked ticker should always be selected"
    assert "B" not in result, "B is too correlated with A and should be skipped"
    assert "C" in result, "C is uncorrelated and should be selected"


def test_correlation_filter_all_uncorrelated():
    """When all tickers are uncorrelated, all should be selected up to top_n."""
    np.random.seed(42)
    n_days = 100
    dates = pd.bdate_range("2023-01-02", periods=n_days)

    data = {}
    for i in range(4):
        data[f"T{i}"] = 100 * np.exp(np.cumsum(np.random.normal(0, 0.02, n_days)))
    prices = pd.DataFrame(data, index=dates)

    result = _greedy_correlation_filter(
        ranked_tickers=["T0", "T1", "T2", "T3"],
        prices_window=prices,
        top_n=4,
        correlation_threshold=0.65,
        correlation_lookback=60,
    )
    assert len(result) == 4


def test_correlation_filter_respects_top_n():
    """Never return more than top_n tickers."""
    np.random.seed(42)
    n_days = 100
    dates = pd.bdate_range("2023-01-02", periods=n_days)

    data = {}
    for i in range(5):
        data[f"T{i}"] = 100 * np.exp(np.cumsum(np.random.normal(0, 0.02, n_days)))
    prices = pd.DataFrame(data, index=dates)

    result = _greedy_correlation_filter(
        ranked_tickers=["T0", "T1", "T2", "T3", "T4"],
        prices_window=prices,
        top_n=2,
        correlation_threshold=0.99,
        correlation_lookback=60,
    )
    assert len(result) <= 2


def test_correlation_filter_empty_input():
    """Empty ranked_tickers returns empty list."""
    prices = pd.DataFrame()
    result = _greedy_correlation_filter(
        [], prices, top_n=5, correlation_threshold=0.65, correlation_lookback=60,
    )
    assert result == []


def test_correlation_filter_insufficient_data_falls_back():
    """When less than 10 rows of return data, filter is skipped (fallback to top_n)."""
    dates = pd.bdate_range("2023-01-02", periods=5)
    data = {"A": [100, 101, 102, 103, 104], "B": [100, 99, 98, 97, 96]}
    prices = pd.DataFrame(data, index=dates)

    result = _greedy_correlation_filter(
        ranked_tickers=["A", "B"],
        prices_window=prices,
        top_n=2,
        correlation_threshold=0.1,  # very strict — would filter everything if data existed
        correlation_lookback=60,
    )
    assert result == ["A", "B"], "Should fall back to top_n slice when insufficient data"


def test_correlation_filter_disabled_matches_original(simple_prices, simple_tickers, kama_low):
    """With enable_correlation_filter=False, output matches original behavior."""
    result_original = get_buy_candidates(
        simple_prices, simple_tickers, kama_low, enable_correlation_filter=False,
    )
    result_disabled = get_buy_candidates(
        simple_prices, simple_tickers, kama_low, enable_correlation_filter=False,
        correlation_threshold=0.1,
    )
    assert result_original == result_disabled
