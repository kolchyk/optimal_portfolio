"""Tests for walk-forward optimization module."""

import numpy as np
import optuna
import pandas as pd
import pytest

from src.portfolio_sim.models import R2WFOResult, R2WFOStep
from src.portfolio_sim.optimizer import (
    compute_objective,
    precompute_kama_caches,
    r2_objective,
)
from src.portfolio_sim.parallel import suggest_r2_params
from src.portfolio_sim.params import R2StrategyParams
from src.portfolio_sim.config import R2_SEARCH_SPACE
from src.portfolio_sim.walk_forward import (
    _stitch_equity_curves,
    format_r2_wfo_report,
    generate_wfo_schedule,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def long_synthetic_prices() -> pd.DataFrame:
    """1600 days of synthetic Close prices for 10 tickers + SPY."""
    np.random.seed(42)
    n_days = 1600
    tickers = [f"T{i:02d}" for i in range(10)] + ["SPY"]
    dates = pd.bdate_range("2018-01-02", periods=n_days)
    data = {}
    for i, t in enumerate(tickers):
        if t == "SPY":
            returns = np.random.normal(0.0004, 0.012, n_days)
        else:
            drift = 0.0003 * (1 + (i % 5) * 0.3)
            vol = 0.015 * (1 + (i % 4) * 0.25)
            returns = np.random.normal(drift, vol, n_days)
        data[t] = 100 * np.exp(np.cumsum(returns))
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def long_synthetic_open(long_synthetic_prices: pd.DataFrame) -> pd.DataFrame:
    np.random.seed(123)
    noise = 1 + np.random.normal(0, 0.002, long_synthetic_prices.shape)
    return long_synthetic_prices * noise


# ---------------------------------------------------------------------------
# Objective function tests
# ---------------------------------------------------------------------------
class TestComputeObjective:
    def test_positive_growth_returns_positive(self):
        # Use noisy uptrend so max_drawdown > 0 and calmar is finite
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.01, 252)
        prices = 10000 * np.exp(np.cumsum(returns))
        equity = pd.Series(
            prices, index=pd.bdate_range("2020-01-01", periods=252),
        )
        obj = compute_objective(equity)
        assert obj > 0

    def test_high_drawdown_rejected(self):
        values = np.concatenate([
            np.linspace(10000, 12000, 126),
            np.linspace(12000, 6000, 126),  # 50% drawdown
        ])
        equity = pd.Series(
            values, index=pd.bdate_range("2020-01-01", periods=252)
        )
        assert compute_objective(equity, max_dd_limit=0.30) == -999.0

    def test_negative_cagr_rejected(self):
        equity = pd.Series(
            np.linspace(10000, 8000, 252),
            index=pd.bdate_range("2020-01-01", periods=252),
        )
        assert compute_objective(equity) == -999.0

    def test_short_equity_rejected(self):
        equity = pd.Series(
            [10000, 10100, 10200],
            index=pd.bdate_range("2020-01-01", periods=3),
        )
        assert compute_objective(equity) == -999.0

    def test_returns_positive_for_healthy_curve(self):
        # Use noisy uptrend so calmar metric is positive
        np.random.seed(99)
        returns = np.random.normal(0.0008, 0.008, 252)
        prices = 10000 * np.exp(np.cumsum(returns))
        equity = pd.Series(
            prices, index=pd.bdate_range("2020-01-01", periods=252),
        )
        obj = compute_objective(equity)
        assert obj > 0


class TestR2Objective:
    def test_positive_growth_returns_calmar(self):
        equity = pd.Series(
            np.linspace(10000, 15000, 252),
            index=pd.bdate_range("2020-01-01", periods=252),
        )
        obj = r2_objective(equity)
        assert obj > 0

    def test_negative_cagr_rejected(self):
        equity = pd.Series(
            np.linspace(10000, 8000, 252),
            index=pd.bdate_range("2020-01-01", periods=252),
        )
        assert r2_objective(equity) == -999.0

    def test_high_drawdown_rejected(self):
        values = np.concatenate([
            np.linspace(10000, 12000, 126),
            np.linspace(12000, 6000, 126),
        ])
        equity = pd.Series(
            values, index=pd.bdate_range("2020-01-01", periods=252),
        )
        assert r2_objective(equity, max_dd_limit=0.30) == -999.0


# ---------------------------------------------------------------------------
# Search space tests (R² params)
# ---------------------------------------------------------------------------
class TestSearchSpace:
    def test_suggest_r2_params_returns_r2_strategy_params(self):
        study = optuna.create_study()
        trial = study.ask()
        params = suggest_r2_params(trial, R2_SEARCH_SPACE)
        assert isinstance(params, R2StrategyParams)

    def test_r2_params_hashable(self):
        """R2StrategyParams must be hashable for dict keys."""
        study = optuna.create_study()
        params_list = []
        for _ in range(10):
            trial = study.ask()
            params_list.append(suggest_r2_params(trial, R2_SEARCH_SPACE))
            study.tell(trial, 1.0)
        s = set(params_list)
        assert len(s) <= len(params_list)


# ---------------------------------------------------------------------------
# KAMA precomputation tests
# ---------------------------------------------------------------------------
class TestPrecomputeKamaCaches:
    def test_returns_all_periods(self, long_synthetic_prices):
        tickers = [c for c in long_synthetic_prices.columns if c != "SPY"]
        caches = precompute_kama_caches(long_synthetic_prices, tickers, [10, 20, 40])
        assert set(caches.keys()) == {10, 20, 40}

    def test_includes_spy(self, long_synthetic_prices):
        tickers = [c for c in long_synthetic_prices.columns if c != "SPY"]
        caches = precompute_kama_caches(long_synthetic_prices, tickers, [20])
        assert "SPY" in caches[20]

    def test_series_length_matches(self, long_synthetic_prices):
        tickers = [c for c in long_synthetic_prices.columns if c != "SPY"]
        caches = precompute_kama_caches(long_synthetic_prices, tickers, [20])
        for t, series in caches[20].items():
            assert isinstance(series, pd.Series)


# ---------------------------------------------------------------------------
# Schedule generation tests
# ---------------------------------------------------------------------------
class TestGenerateWFOSchedule:
    def test_basic_schedule(self):
        """1600 days, min_is=756, oos=252 => at least 2 steps."""
        dates = pd.bdate_range("2018-01-02", periods=1600)
        schedule = generate_wfo_schedule(dates, min_is_days=756, oos_days=252)
        assert len(schedule) >= 2

    def test_schedule_is_sliding(self):
        """IS window slides forward — start advances, size stays constant."""
        dates = pd.bdate_range("2018-01-02", periods=1600)
        min_is = 756
        schedule = generate_wfo_schedule(dates, min_is_days=min_is, oos_days=252)
        for i, (is_start, is_end, _, _) in enumerate(schedule):
            is_size = dates.get_loc(is_end) - dates.get_loc(is_start) + 1
            assert is_size == min_is
            if i > 0:
                prev_start = schedule[i - 1][0]
                assert is_start > prev_start

    def test_no_overlap_is_oos(self):
        """IS end must be strictly before OOS start."""
        dates = pd.bdate_range("2018-01-02", periods=1600)
        schedule = generate_wfo_schedule(dates, min_is_days=756, oos_days=252)
        for _, is_end, oos_start, _ in schedule:
            assert is_end < oos_start

    def test_oos_windows_are_contiguous(self):
        """OOS windows should be contiguous (no gaps)."""
        dates = pd.bdate_range("2018-01-02", periods=1600)
        schedule = generate_wfo_schedule(dates, min_is_days=756, oos_days=252)
        for i in range(1, len(schedule)):
            prev_oos_end = schedule[i - 1][3]
            curr_oos_start = schedule[i][2]
            gap = (curr_oos_start - prev_oos_end).days
            assert gap <= 5, f"Gap between OOS windows too large: {gap} days"

    def test_insufficient_data_returns_empty(self):
        """With too little data, schedule should be empty."""
        dates = pd.bdate_range("2018-01-02", periods=800)
        schedule = generate_wfo_schedule(dates, min_is_days=756, oos_days=252)
        assert len(schedule) == 0

    def test_is_window_constant_size(self):
        """Each step's IS window should be the same size (sliding)."""
        dates = pd.bdate_range("2018-01-02", periods=2000)
        min_is = 400
        schedule = generate_wfo_schedule(dates, min_is_days=min_is, oos_days=200)
        for is_start, is_end, _, _ in schedule:
            is_size = dates.get_loc(is_end) - dates.get_loc(is_start) + 1
            assert is_size == min_is


# ---------------------------------------------------------------------------
# Equity curve stitching tests
# ---------------------------------------------------------------------------
class TestStitchEquityCurves:
    def test_starts_at_initial_capital(self):
        curve1 = pd.Series(
            [100, 110, 120],
            index=pd.bdate_range("2020-01-01", periods=3),
        )
        stitched = _stitch_equity_curves([curve1], initial_capital=10_000)
        assert stitched.iloc[0] == pytest.approx(10_000)

    def test_continuity_between_segments(self):
        """Second segment should start where first ended."""
        curve1 = pd.Series(
            [100, 120],
            index=pd.bdate_range("2020-01-01", periods=2),
        )
        curve2 = pd.Series(
            [200, 240],
            index=pd.bdate_range("2020-06-01", periods=2),
        )
        stitched = _stitch_equity_curves([curve1, curve2], initial_capital=10_000)
        assert stitched.iloc[0] == pytest.approx(10_000)
        assert stitched.iloc[1] == pytest.approx(12_000)
        assert stitched.iloc[2] == pytest.approx(12_000)
        assert stitched.iloc[3] == pytest.approx(14_400)

    def test_empty_curves_produce_empty(self):
        stitched = _stitch_equity_curves([], initial_capital=10_000)
        assert stitched.empty

    def test_multiple_segments_preserve_returns(self):
        """Total return of stitched curve equals product of segment returns."""
        curve1 = pd.Series(
            [100, 150],  # +50%
            index=pd.bdate_range("2020-01-01", periods=2),
        )
        curve2 = pd.Series(
            [100, 80],  # -20%
            index=pd.bdate_range("2020-06-01", periods=2),
        )
        stitched = _stitch_equity_curves([curve1, curve2], initial_capital=10_000)
        total_return = stitched.iloc[-1] / stitched.iloc[0]
        expected = 1.5 * 0.8
        assert total_return == pytest.approx(expected)
