"""Tests for the walk-forward parameter optimizer."""

import numpy as np
import pandas as pd
import pytest

from src.portfolio_sim.config import INITIAL_CAPITAL
from src.portfolio_sim.optimizer import (
    WalkForwardConfig,
    build_param_grid,
    compute_is_oos_degradation,
    compute_objective,
    compute_stability_scores,
    concatenate_oos_equity,
    generate_folds,
    precompute_kama_caches,
    FoldResult,
    analyze_parameter_stability,
)
from src.portfolio_sim.params import StrategyParams


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def long_dates() -> pd.DatetimeIndex:
    """1600 business days (~6.3 years) — enough for multiple folds."""
    return pd.bdate_range("2018-01-02", periods=1600)


@pytest.fixture
def short_dates() -> pd.DatetimeIndex:
    """500 business days — not enough for 2 folds with default config."""
    return pd.bdate_range("2021-01-04", periods=500)


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
# Fold generation tests
# ---------------------------------------------------------------------------
class TestGenerateFolds:
    def test_minimum_two_folds(self, long_dates):
        config = WalkForwardConfig(min_is_days=756, oos_days=252, step_days=252)
        folds = generate_folds(long_dates, config)
        assert len(folds) >= 2

    def test_too_few_dates_raises(self, short_dates):
        config = WalkForwardConfig(min_is_days=756, oos_days=252, step_days=252)
        with pytest.raises(ValueError, match="at least 2"):
            generate_folds(short_dates, config)

    def test_oos_starts_after_is_ends(self, long_dates):
        config = WalkForwardConfig(min_is_days=756, oos_days=252, step_days=252)
        folds = generate_folds(long_dates, config)
        for is_dates, oos_dates in folds:
            assert oos_dates[0] > is_dates[-1], "OOS must start after IS ends"

    def test_is_window_expands(self, long_dates):
        config = WalkForwardConfig(min_is_days=756, oos_days=252, step_days=252)
        folds = generate_folds(long_dates, config)
        is_lengths = [len(is_d) for is_d, _ in folds]
        for i in range(1, len(is_lengths)):
            assert is_lengths[i] > is_lengths[i - 1], "IS window should expand"

    def test_oos_window_fixed_length(self, long_dates):
        config = WalkForwardConfig(min_is_days=756, oos_days=252, step_days=252)
        folds = generate_folds(long_dates, config)
        for _, oos_dates in folds:
            assert len(oos_dates) == 252

    def test_no_overlap_between_folds_oos(self, long_dates):
        config = WalkForwardConfig(min_is_days=756, oos_days=252, step_days=252)
        folds = generate_folds(long_dates, config)
        for i in range(1, len(folds)):
            prev_oos_end = folds[i - 1][1][-1]
            curr_oos_start = folds[i][1][0]
            assert curr_oos_start > prev_oos_end or curr_oos_start == folds[i][1][0]


# ---------------------------------------------------------------------------
# Objective function tests
# ---------------------------------------------------------------------------
class TestComputeObjective:
    def test_positive_growth_returns_positive(self):
        equity = pd.Series(
            np.linspace(10000, 15000, 252),
            index=pd.bdate_range("2020-01-01", periods=252),
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

    def test_drawdown_floor_applied(self):
        """Even with tiny drawdown, floor at 5% prevents inflated Calmar."""
        equity = pd.Series(
            np.linspace(10000, 12000, 252),
            index=pd.bdate_range("2020-01-01", periods=252),
        )
        obj = compute_objective(equity)
        # With 5% floor and ~20% return, Calmar should be reasonable
        assert obj < 10


# ---------------------------------------------------------------------------
# OOS concatenation tests
# ---------------------------------------------------------------------------
class TestConcatenateOOS:
    def _make_fold_result(self, fold_idx, start_val, end_val, n_days, start_date):
        dates = pd.bdate_range(start_date, periods=n_days)
        values = np.linspace(start_val, end_val, n_days)
        equity = pd.Series(values, index=dates)
        return FoldResult(
            fold_index=fold_idx,
            is_start=dates[0],
            is_end=dates[0],
            oos_start=dates[0],
            oos_end=dates[-1],
            best_params=StrategyParams(),
            is_objective=1.0,
            oos_equity=equity,
            oos_metrics={},
        )

    def test_single_fold_unchanged(self):
        fr = self._make_fold_result(0, 10000, 12000, 252, "2020-01-01")
        result = concatenate_oos_equity([fr])
        pd.testing.assert_series_equal(result, fr.oos_equity)

    def test_multi_fold_continuity(self):
        fr1 = self._make_fold_result(0, 10000, 12000, 100, "2020-01-01")
        fr2 = self._make_fold_result(1, 10000, 11000, 100, "2020-06-01")
        result = concatenate_oos_equity([fr1, fr2])
        assert len(result) == 200
        # Second segment should start where first ended
        assert abs(result.iloc[100] - 12000) < 1

    def test_empty_returns_empty(self):
        result = concatenate_oos_equity([])
        assert result.empty


# ---------------------------------------------------------------------------
# Parameter grid tests
# ---------------------------------------------------------------------------
class TestBuildParamGrid:
    def test_default_grid_size(self):
        params = build_param_grid()
        assert len(params) == 5 * 5 * 5 * 5  # 625

    def test_custom_grid(self):
        grid = {
            "kama_period": [10, 20],
            "lookback_period": [60],
            "kama_buffer": [0.01],
            "top_n": [20],
        }
        params = build_param_grid(grid)
        assert len(params) == 2

    def test_all_params_are_strategy_params(self):
        params = build_param_grid()
        for p in params:
            assert isinstance(p, StrategyParams)

    def test_params_hashable(self):
        """StrategyParams must be hashable for dict keys."""
        params = build_param_grid()
        s = set(params)
        assert len(s) == len(params)


# ---------------------------------------------------------------------------
# Stability analysis tests
# ---------------------------------------------------------------------------
class TestParameterStability:
    def _make_fold_results(self, params_list):
        results = []
        for i, bp in enumerate(params_list):
            dates = pd.bdate_range("2020-01-01", periods=10)
            eq = pd.Series(np.linspace(10000, 11000, 10), index=dates)
            results.append(
                FoldResult(
                    fold_index=i,
                    is_start=dates[0],
                    is_end=dates[-1],
                    oos_start=dates[0],
                    oos_end=dates[-1],
                    best_params=bp,
                    is_objective=1.5,
                    oos_equity=eq,
                    oos_metrics={"cagr": 0.1, "max_drawdown": 0.05, "calmar": 2.0},
                )
            )
        return results

    def test_stable_params_high_score(self):
        bp = StrategyParams(kama_period=20, lookback_period=60, top_n=20, kama_buffer=0.01)
        folds = self._make_fold_results([bp, bp, bp])
        df = analyze_parameter_stability(folds)
        scores = compute_stability_scores(df)
        for name, score in scores.items():
            assert score == 1.0

    def test_unstable_params_low_score(self):
        folds = self._make_fold_results([
            StrategyParams(kama_period=10),
            StrategyParams(kama_period=40),
            StrategyParams(kama_period=10),
        ])
        df = analyze_parameter_stability(folds)
        scores = compute_stability_scores(df)
        assert scores["kama_period"] == 0.0  # max range covered


class TestISOOSDegradation:
    def test_no_degradation(self):
        dates = pd.bdate_range("2020-01-01", periods=10)
        eq = pd.Series(np.linspace(10000, 11000, 10), index=dates)
        fr = FoldResult(
            fold_index=0,
            is_start=dates[0],
            is_end=dates[-1],
            oos_start=dates[0],
            oos_end=dates[-1],
            best_params=StrategyParams(),
            is_objective=2.0,
            oos_equity=eq,
            oos_metrics={"calmar": 2.0},
        )
        degradations = compute_is_oos_degradation([fr])
        assert degradations[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# KAMA precomputation test
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
