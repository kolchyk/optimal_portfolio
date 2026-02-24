"""Tests for alpha strategy (Long/Short Equity)."""

import numpy as np
import pandas as pd
import pytest

from src.portfolio_sim.config import MAX_GROSS_EXPOSURE, MAX_WEIGHT, SAFE_HAVEN_TICKER, StrategyParams
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
    """KAMA values set low so all tickers pass the bull KAMA filter (Close > KAMA)."""
    return {t: 0.0 for t in simple_tickers}


@pytest.fixture
def kama_below(simple_prices, simple_tickers):
    """KAMA values set very high so all tickers FAIL the bull KAMA filter."""
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


def test_kama_filter_removes_all_bull(simple_prices, simple_tickers, kama_below):
    """When all tickers fail bull KAMA filter, weights should be all zeros."""
    params = StrategyParams()
    weights = compute_target_weights(simple_prices, simple_tickers, params, kama_below, is_bull=True)
    assert np.allclose(weights, 0.0)


def test_kama_filter_removes_all_bear(simple_prices, simple_tickers, kama_above):
    """When all tickers pass bull filter (Close > KAMA), they fail bear filter."""
    params = StrategyParams()
    weights = compute_target_weights(simple_prices, simple_tickers, params, kama_above, is_bull=False)
    assert np.allclose(weights, 0.0)


def test_shv_never_gets_weight(simple_prices, simple_tickers, kama_above):
    """SHV should never be selected by alpha (engine handles SHV parking)."""
    params = StrategyParams(top_n_selection=15)
    weights = compute_target_weights(simple_prices, simple_tickers, params, kama_above)
    shv_idx = simple_tickers.index(SAFE_HAVEN_TICKER)
    assert weights[shv_idx] == 0.0


def test_downtrend_tickers_excluded_in_bull(simple_prices, simple_tickers, kama_above):
    """In bull mode, tickers with negative momentum should get zero weight."""
    params = StrategyParams(top_n_selection=5)
    weights = compute_target_weights(simple_prices, simple_tickers, params, kama_above, is_bull=True)
    # T05-T09 have negative momentum
    for i in range(5, 10):
        idx = simple_tickers.index(f"T{i:02d}")
        assert weights[idx] == 0.0


def test_bull_mode_no_negative_weights(simple_prices, simple_tickers, kama_above):
    """In bull mode, no weight should be negative."""
    params = StrategyParams(top_n_selection=5)
    weights = compute_target_weights(simple_prices, simple_tickers, params, kama_above, is_bull=True)
    assert np.all(weights >= 0)


def test_bear_mode_negative_weights(simple_prices, simple_tickers):
    """In bear mode, alpha weights should be <= 0 for selected tickers."""
    # KAMA set high so all tickers have Close < KAMA (pass bear filter)
    kama_high = {t: 1e6 for t in simple_tickers}
    params = StrategyParams(top_n_selection=5)
    weights = compute_target_weights(simple_prices, simple_tickers, params, kama_high, is_bull=False)
    # T05-T09 have negative momentum -> should be shorted (negative weights)
    has_negative = any(w < 0 for w in weights)
    assert has_negative


def test_bear_mode_no_positive_alpha_weights(simple_prices, simple_tickers):
    """In bear mode, no alpha position should have positive weight."""
    kama_high = {t: 1e6 for t in simple_tickers}
    params = StrategyParams(top_n_selection=5)
    weights = compute_target_weights(simple_prices, simple_tickers, params, kama_high, is_bull=False)
    for i, t in enumerate(simple_tickers):
        if t != SAFE_HAVEN_TICKER:
            assert weights[i] <= 0


def test_gross_exposure_capped(simple_prices, simple_tickers, kama_above):
    """Sum of absolute weights must not exceed MAX_GROSS_EXPOSURE."""
    params = StrategyParams(top_n_selection=15)
    weights = compute_target_weights(simple_prices, simple_tickers, params, kama_above, is_bull=True)
    assert np.abs(weights).sum() <= MAX_GROSS_EXPOSURE + 1e-6


def test_gross_exposure_capped_bear(simple_prices, simple_tickers):
    """Sum of absolute weights must not exceed MAX_GROSS_EXPOSURE in bear mode."""
    kama_high = {t: 1e6 for t in simple_tickers}
    params = StrategyParams(top_n_selection=15)
    weights = compute_target_weights(simple_prices, simple_tickers, params, kama_high, is_bull=False)
    assert np.abs(weights).sum() <= MAX_GROSS_EXPOSURE + 1e-6


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

    params = StrategyParams(top_n_selection=5)
    weights = compute_target_weights(prices, tickers, params, kama, is_bull=True)

    # A (low vol) should have higher weight than B (high vol)
    assert weights[0] > weights[1]
