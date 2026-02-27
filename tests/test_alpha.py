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


def test_ranked_descending_by_raw_momentum(
    simple_prices, simple_tickers, kama_low,
):
    """When risk-adjusted is off, candidates sorted by raw momentum descending."""
    result = get_buy_candidates(
        simple_prices, simple_tickers, kama_low, use_risk_adjusted=False,
    )
    if len(result) >= 2:
        for i in range(len(result) - 1):
            t_a, t_b = result[i], result[i + 1]
            mom_a = simple_prices[t_a].iloc[-1] / simple_prices[t_a].iloc[0] - 1
            mom_b = simple_prices[t_b].iloc[-1] / simple_prices[t_b].iloc[0] - 1
            assert mom_a >= mom_b


def test_ranked_descending_by_er_adjusted(
    simple_prices, simple_tickers, kama_low,
):
    """When risk-adjusted is on, candidates sorted by return * ER² descending."""
    result = get_buy_candidates(
        simple_prices, simple_tickers, kama_low, use_risk_adjusted=True,
    )
    assert len(result) >= 2
    # Compute ER-adjusted scores independently
    scores = {}
    for t in result:
        series = simple_prices[t].dropna()
        raw_ret = series.iloc[-1] / series.iloc[0] - 1.0
        price_change = abs(series.iloc[-1] - series.iloc[0])
        vol = series.diff().abs().iloc[1:].sum()
        er = min(price_change / vol, 1.0) if vol > 1e-8 else 0.0
        scores[t] = raw_ret * (er ** 2)
    for i in range(len(result) - 1):
        assert scores[result[i]] >= scores[result[i + 1]]


def test_empty_tickers_returns_empty(simple_prices, kama_low):
    result = get_buy_candidates(simple_prices, [], kama_low)
    assert result == []


def test_er_penalizes_choppy_action():
    """A choppy ticker with same return should score lower than a smooth one."""
    dates = pd.bdate_range("2023-01-02", periods=50)
    # Smooth uptrend: monotonically increasing
    smooth = pd.Series(np.linspace(100, 120, 50), index=dates)
    # Choppy: same start/end but oscillates
    choppy_base = np.linspace(100, 120, 50)
    choppy = pd.Series(
        choppy_base + 5 * np.sin(np.linspace(0, 20, 50)), index=dates,
    )
    # Force same endpoint
    choppy.iloc[-1] = 120.0

    prices = pd.DataFrame({"SMOOTH": smooth, "CHOPPY": choppy}, index=dates)
    kama_low = {"SMOOTH": 0.0, "CHOPPY": 0.0}

    result = get_buy_candidates(
        prices, ["SMOOTH", "CHOPPY"], kama_low, use_risk_adjusted=True,
    )
    assert result[0] == "SMOOTH"


def test_precomputed_er_scoring(simple_prices, simple_tickers, kama_low):
    """Precomputed ER path should produce same ranking as non-precomputed."""
    result_basic = get_buy_candidates(
        simple_prices, simple_tickers, kama_low, use_risk_adjusted=True,
    )
    # Compute precomputed values matching non-precomputed logic
    momentum = pd.Series({
        t: simple_prices[t].iloc[-1] / simple_prices[t].iloc[0] - 1.0
        for t in simple_tickers
    })
    er_values = {}
    for t in simple_tickers:
        s = simple_prices[t].dropna()
        pc = abs(s.iloc[-1] - s.iloc[0])
        vol = s.diff().abs().iloc[1:].sum()
        er_values[t] = min(pc / vol, 1.0) if vol > 1e-8 else 0.0
    er = pd.Series(er_values)

    result_precomputed = get_buy_candidates(
        simple_prices.iloc[[-1]], simple_tickers, kama_low,
        use_risk_adjusted=True,
        precomputed_momentum=momentum,
        precomputed_er=er,
    )
    assert result_basic == result_precomputed
