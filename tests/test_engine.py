"""Tests for R² Momentum backtest engine."""

import numpy as np
import pandas as pd
import pytest

from src.portfolio_sim.config import INITIAL_CAPITAL
from src.portfolio_sim.engine import (
    COST_RATE,
    compute_atr_close,
    compute_r2_momentum,
    has_large_gap,
    is_above_kama,
    is_risk_on,
    run_backtest,
    select_r2_assets,
)
from src.portfolio_sim.indicators import compute_kama_series


@pytest.fixture
def sim_data():
    """250 days of full history for 10 tickers + SPY."""
    np.random.seed(42)
    n_full = 250
    dates = pd.bdate_range("2022-01-03", periods=n_full)
    tickers = [f"T{i:02d}" for i in range(10)]

    data = {}
    for t in tickers:
        data[t] = 100 * np.exp(
            np.cumsum(np.random.normal(0.0003, 0.015, n_full))
        )
    data["SPY"] = 100 * np.exp(
        np.cumsum(np.random.normal(0.0004, 0.01, n_full))
    )

    close = pd.DataFrame(data, index=dates)
    open_ = close * (1 + np.random.normal(0, 0.001, close.shape))
    return close, open_, tickers


# ---------------------------------------------------------------------------
# run_backtest integration tests
# ---------------------------------------------------------------------------
def test_returns_equity_and_spy(sim_data):
    close, open_, tickers = sim_data
    equity, spy_eq, holdings, cash, trades = run_backtest(
        close, open_, tickers, INITIAL_CAPITAL,
    )
    assert isinstance(equity, pd.Series)
    assert isinstance(spy_eq, pd.Series)
    assert len(equity) == len(spy_eq)


def test_equity_starts_near_initial_capital(sim_data):
    close, open_, tickers = sim_data
    equity, _, _, _, _ = run_backtest(close, open_, tickers, INITIAL_CAPITAL)
    assert abs(equity.iloc[0] - INITIAL_CAPITAL) / INITIAL_CAPITAL < 0.10


def test_spy_benchmark_starts_at_initial_capital(sim_data):
    close, open_, tickers = sim_data
    _, spy_eq, _, _, _ = run_backtest(close, open_, tickers, INITIAL_CAPITAL)
    assert spy_eq.iloc[0] == pytest.approx(INITIAL_CAPITAL, rel=0.01)


def test_equity_positive(sim_data):
    close, open_, tickers = sim_data
    equity, _, _, _, _ = run_backtest(close, open_, tickers, INITIAL_CAPITAL)
    assert (equity >= 0).all()


def test_holdings_history_is_dataframe(sim_data):
    close, open_, tickers = sim_data
    _, _, holdings, _, _ = run_backtest(close, open_, tickers, INITIAL_CAPITAL)
    assert isinstance(holdings, pd.DataFrame)


def test_cash_history_is_series(sim_data):
    close, open_, tickers = sim_data
    _, _, _, cash, _ = run_backtest(close, open_, tickers, INITIAL_CAPITAL)
    assert isinstance(cash, pd.Series)
    assert len(cash) == len(close.index) - (max(90, 40) + 10)  # warmup trimmed


def test_trade_log_is_list(sim_data):
    close, open_, tickers = sim_data
    _, _, _, _, trades = run_backtest(close, open_, tickers, INITIAL_CAPITAL)
    assert isinstance(trades, list)


def test_empty_on_insufficient_data():
    """With fewer rows than warmup, should return empty series."""
    dates = pd.bdate_range("2022-01-03", periods=10)
    close = pd.DataFrame(
        {"SPY": np.linspace(100, 105, 10), "A": np.linspace(50, 55, 10)},
        index=dates,
    )
    open_ = close * 1.001
    equity, spy_eq, _, _, _ = run_backtest(
        close, open_, ["A"], INITIAL_CAPITAL, warmup_days=20,
    )
    assert equity.empty


# ---------------------------------------------------------------------------
# Utility function unit tests
# ---------------------------------------------------------------------------
class TestIsRiskOn:
    def test_risk_on_above_kama(self):
        dates = pd.bdate_range("2022-01-03", periods=5)
        spy_kama = pd.Series([100, 100, 100, 100, 100], index=dates, dtype=float)
        spy_close = pd.Series([105, 105, 105, 105, 105], index=dates, dtype=float)
        assert is_risk_on(spy_kama, spy_close, dates[2])

    def test_risk_off_below_kama(self):
        dates = pd.bdate_range("2022-01-03", periods=5)
        spy_kama = pd.Series([100, 100, 100, 100, 100], index=dates, dtype=float)
        spy_close = pd.Series([90, 90, 90, 90, 90], index=dates, dtype=float)
        assert not is_risk_on(spy_kama, spy_close, dates[2])

    def test_nan_kama_assumes_risk_on(self):
        dates = pd.bdate_range("2022-01-03", periods=5)
        spy_kama = pd.Series([np.nan] * 5, index=dates, dtype=float)
        spy_close = pd.Series([100] * 5, index=dates, dtype=float)
        assert is_risk_on(spy_kama, spy_close, dates[2])


class TestIsAboveKama:
    def test_above(self):
        dates = pd.bdate_range("2022-01-03", periods=5)
        kama = pd.Series([100] * 5, index=dates, dtype=float)
        prices = pd.Series([110] * 5, index=dates, dtype=float)
        assert is_above_kama(kama, prices, dates[2])

    def test_below(self):
        dates = pd.bdate_range("2022-01-03", periods=5)
        kama = pd.Series([100] * 5, index=dates, dtype=float)
        prices = pd.Series([90] * 5, index=dates, dtype=float)
        assert not is_above_kama(kama, prices, dates[2])


class TestHasLargeGap:
    def test_no_gap(self):
        dates = pd.bdate_range("2022-01-03", periods=100)
        prices = pd.Series(np.linspace(100, 110, 100), index=dates)
        assert has_large_gap(prices, dates[-1], lookback=90) is False

    def test_with_gap(self):
        dates = pd.bdate_range("2022-01-03", periods=100)
        values = np.linspace(100, 110, 100)
        values[50] = values[49] * 1.20  # 20% gap
        prices = pd.Series(values, index=dates)
        assert has_large_gap(prices, dates[-1], lookback=90, threshold=0.15) is True


class TestComputeAtrClose:
    def test_atr_computed(self):
        dates = pd.bdate_range("2022-01-03", periods=30)
        prices = pd.Series(np.linspace(100, 130, 30), index=dates)
        atr = compute_atr_close(prices, dates[-1], period=20)
        assert atr > 0
        assert not np.isnan(atr)

    def test_atr_nan_insufficient_data(self):
        dates = pd.bdate_range("2022-01-03", periods=5)
        prices = pd.Series(np.linspace(100, 105, 5), index=dates)
        atr = compute_atr_close(prices, dates[-1], period=20)
        assert np.isnan(atr)


class TestComputeR2Momentum:
    def test_uptrend_positive_score(self):
        prices = pd.Series(np.exp(np.linspace(0, 1, 100)))
        ann_ret, r2, score = compute_r2_momentum(prices, period=90)
        assert ann_ret > 0
        assert 0 < r2 <= 1.0
        assert score > 0

    def test_insufficient_data_returns_nan(self):
        prices = pd.Series([100, 101, 102])
        ann_ret, r2, score = compute_r2_momentum(prices, period=90)
        assert np.isnan(ann_ret)
        assert np.isnan(r2)
        assert np.isnan(score)

    def test_high_r2_for_linear_trend(self):
        prices = pd.Series(np.exp(np.linspace(0, 0.5, 100)))
        _, r2, _ = compute_r2_momentum(prices, period=90)
        assert r2 > 0.95  # nearly perfect linear trend in log space
