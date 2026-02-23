"""Tests for alpha strategy (Simple Baseline)."""

import numpy as np
import pandas as pd
import pytest

from src.portfolio_sim.config import MAX_WEIGHT, SAFE_HAVEN_TICKER, StrategyParams
from src.portfolio_sim.alpha import compute_target_weights


@pytest.fixture
def simple_prices():
    """10 tickers + SHV, 200 days, with clear momentum differences."""
    np.random.seed(42)
    n_days = 200
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    data = {}

    # T00-T04: strong uptrend (good momentum)
    for i in range(5):
        drift = 0.001 * (5 - i)
        data[f"T{i:02d}"] = 100 * np.exp(np.cumsum(np.random.normal(drift, 0.01, n_days)))

    # T05-T09: downtrend (bad momentum)
    for i in range(5, 10):
        data[f"T{i:02d}"] = 100 * np.exp(np.cumsum(np.random.normal(-0.001, 0.01, n_days)))

    data[SAFE_HAVEN_TICKER] = np.linspace(100, 100.5, n_days)
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def simple_tickers(simple_prices):
    return list(simple_prices.columns)


@pytest.fixture
def kama_above(simple_prices, simple_tickers):
    """KAMA values set low so all tickers pass the KAMA filter."""
    return {t: 0.0 for t in simple_tickers}


@pytest.fixture
def kama_below(simple_prices, simple_tickers):
    """KAMA values set very high so all tickers FAIL the KAMA filter."""
    return {t: 1e6 for t in simple_tickers}


def test_returns_correct_shape(simple_prices, simple_tickers, kama_above):
    params = StrategyParams(top_n_selection=5)
    weights = compute_target_weights(simple_prices, simple_tickers, params, kama_above)
    assert len(weights) == len(simple_tickers)


def test_weights_sum_le_one(simple_prices, simple_tickers, kama_above):
    params = StrategyParams(top_n_selection=5)
    weights = compute_target_weights(simple_prices, simple_tickers, params, kama_above)
    assert weights.sum() <= 1.0 + 1e-6


def test_no_weight_exceeds_max(simple_prices, simple_tickers, kama_above):
    params = StrategyParams(top_n_selection=5)
    weights = compute_target_weights(simple_prices, simple_tickers, params, kama_above)
    assert np.all(weights <= MAX_WEIGHT + 1e-6)


def test_kama_filter_removes_all(simple_prices, simple_tickers, kama_below):
    """When all tickers fail KAMA filter, weights should be all zeros."""
    params = StrategyParams()
    weights = compute_target_weights(simple_prices, simple_tickers, params, kama_below)
    assert np.allclose(weights, 0.0)


def test_shv_never_gets_weight(simple_prices, simple_tickers, kama_above):
    """SHV should never be selected by alpha (engine handles SHV parking)."""
    params = StrategyParams(top_n_selection=15)
    weights = compute_target_weights(simple_prices, simple_tickers, params, kama_above)
    shv_idx = simple_tickers.index(SAFE_HAVEN_TICKER)
    assert weights[shv_idx] == 0.0


def test_downtrend_tickers_excluded(simple_prices, simple_tickers, kama_above):
    """Tickers with negative momentum should get zero weight."""
    params = StrategyParams(top_n_selection=5)
    weights = compute_target_weights(simple_prices, simple_tickers, params, kama_above)
    # T05-T09 have negative momentum
    for i in range(5, 10):
        idx = simple_tickers.index(f"T{i:02d}")
        assert weights[idx] == 0.0


def test_correlation_walkdown():
    """Two perfectly correlated tickers: only one should be selected."""
    np.random.seed(42)
    n_days = 200
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    base_returns = np.random.normal(0.001, 0.01, n_days)

    data = {
        "A": 100 * np.exp(np.cumsum(base_returns)),
        "B": 100 * np.exp(np.cumsum(base_returns * 1.0)),  # Identical returns
        SAFE_HAVEN_TICKER: np.linspace(100, 100.5, n_days),
    }
    prices = pd.DataFrame(data, index=dates)
    tickers = ["A", "B", SAFE_HAVEN_TICKER]
    kama = {"A": 0.0, "B": 0.0}

    params = StrategyParams(max_correlation=0.5, top_n_selection=5)
    weights = compute_target_weights(prices, tickers, params, kama)

    # Both have same momentum, but correlation = 1.0 > 0.5
    # Only one should be selected
    active = [t for t, w in zip(tickers, weights) if w > 0]
    assert len(active) == 1


def test_inverse_vol_weighting():
    """Lower volatility ticker should get higher weight."""
    np.random.seed(42)
    n_days = 200
    dates = pd.bdate_range("2023-01-02", periods=n_days)

    # A: low vol, positive momentum
    # B: high vol, positive momentum
    data = {
        "A": 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.005, n_days))),
        "B": 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.03, n_days))),
        SAFE_HAVEN_TICKER: np.linspace(100, 100.5, n_days),
    }
    prices = pd.DataFrame(data, index=dates)
    tickers = ["A", "B", SAFE_HAVEN_TICKER]
    kama = {"A": 0.0, "B": 0.0}

    params = StrategyParams(max_correlation=0.99, top_n_selection=5)
    weights = compute_target_weights(prices, tickers, params, kama)

    # A (low vol) should have higher weight than B (high vol)
    assert weights[0] > weights[1]
