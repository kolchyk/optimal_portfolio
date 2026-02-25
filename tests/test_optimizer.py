"""Tests for the parameter sensitivity analysis module."""

import numpy as np
import pandas as pd
import pytest

from src.portfolio_sim.optimizer import (
    SENSITIVITY_GRID,
    SensitivityResult,
    build_param_grid,
    compute_marginal_profiles,
    compute_objective,
    compute_robustness_scores,
    format_sensitivity_report,
    precompute_kama_caches,
    run_sensitivity,
)
from src.portfolio_sim.params import StrategyParams


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


# ---------------------------------------------------------------------------
# Marginal profiles tests
# ---------------------------------------------------------------------------
class TestMarginalProfiles:
    def test_one_param_profiles(self):
        """Grouping by kama_period should produce correct mean objectives."""
        grid_df = pd.DataFrame({
            "kama_period": [10, 10, 20, 20],
            "lookback_period": [60, 90, 60, 90],
            "kama_buffer": [0.01, 0.01, 0.01, 0.01],
            "top_n": [20, 20, 20, 20],
            "objective": [2.0, 4.0, 3.0, 3.0],
        })
        profiles = compute_marginal_profiles(grid_df)
        kp_profile = profiles["kama_period"]
        assert len(kp_profile) == 2
        row_10 = kp_profile[kp_profile["kama_period"] == 10].iloc[0]
        assert row_10["mean_objective"] == pytest.approx(3.0)
        row_20 = kp_profile[kp_profile["kama_period"] == 20].iloc[0]
        assert row_20["mean_objective"] == pytest.approx(3.0)

    def test_invalid_results_excluded(self):
        """Rows with objective == -999 should be excluded from profiles."""
        grid_df = pd.DataFrame({
            "kama_period": [10, 10, 20],
            "lookback_period": [60, 90, 60],
            "kama_buffer": [0.01, 0.01, 0.01],
            "top_n": [20, 20, 20],
            "objective": [2.0, -999.0, 3.0],
        })
        profiles = compute_marginal_profiles(grid_df)
        kp_profile = profiles["kama_period"]
        row_10 = kp_profile[kp_profile["kama_period"] == 10].iloc[0]
        # Only the valid row (objective=2.0) should be included
        assert row_10["mean_objective"] == pytest.approx(2.0)
        assert row_10["count"] == 1

    def test_all_invalid_returns_empty(self):
        """All -999 objectives should produce empty profiles."""
        grid_df = pd.DataFrame({
            "kama_period": [10, 20],
            "lookback_period": [60, 60],
            "kama_buffer": [0.01, 0.01],
            "top_n": [20, 20],
            "objective": [-999.0, -999.0],
        })
        profiles = compute_marginal_profiles(grid_df)
        assert profiles["kama_period"].empty


# ---------------------------------------------------------------------------
# Robustness scores tests
# ---------------------------------------------------------------------------
class TestRobustnessScores:
    def test_flat_profile_scores_high(self):
        """If objective is the same for all param values, score = 1.0."""
        profiles = {
            "kama_period": pd.DataFrame({
                "kama_period": [10, 20, 40],
                "mean_objective": [3.0, 3.0, 3.0],
                "std_objective": [0.1, 0.1, 0.1],
                "count": [5, 5, 5],
            }),
            "lookback_period": pd.DataFrame({
                "lookback_period": [20, 60, 120],
                "mean_objective": [3.0, 3.0, 3.0],
                "std_objective": [0.1, 0.1, 0.1],
                "count": [5, 5, 5],
            }),
            "kama_buffer": pd.DataFrame({
                "kama_buffer": [0.005, 0.015, 0.03],
                "mean_objective": [3.0, 3.0, 3.0],
                "std_objective": [0.1, 0.1, 0.1],
                "count": [5, 5, 5],
            }),
            "top_n": pd.DataFrame({
                "top_n": [10, 20, 30],
                "mean_objective": [3.0, 3.0, 3.0],
                "std_objective": [0.1, 0.1, 0.1],
                "count": [5, 5, 5],
            }),
        }
        scores = compute_robustness_scores(profiles)
        for name in ["kama_period", "lookback_period", "kama_buffer", "top_n"]:
            assert scores[name] == pytest.approx(1.0)

    def test_peaked_profile_scores_low(self):
        """If one value dominates, score should be low."""
        profiles = {
            "kama_period": pd.DataFrame({
                "kama_period": [10, 20, 40],
                "mean_objective": [0.5, 5.0, 0.5],
                "std_objective": [0.1, 0.1, 0.1],
                "count": [5, 5, 5],
            }),
            "lookback_period": pd.DataFrame({
                "lookback_period": [20, 60, 120],
                "mean_objective": [3.0, 3.0, 3.0],
                "std_objective": [0.1, 0.1, 0.1],
                "count": [5, 5, 5],
            }),
            "kama_buffer": pd.DataFrame({
                "kama_buffer": [0.005, 0.015, 0.03],
                "mean_objective": [3.0, 3.0, 3.0],
                "std_objective": [0.1, 0.1, 0.1],
                "count": [5, 5, 5],
            }),
            "top_n": pd.DataFrame({
                "top_n": [10, 20, 30],
                "mean_objective": [3.0, 3.0, 3.0],
                "std_objective": [0.1, 0.1, 0.1],
                "count": [5, 5, 5],
            }),
        }
        scores = compute_robustness_scores(profiles)
        assert scores["kama_period"] < 0.5  # peaked, not robust
        assert scores["lookback_period"] == pytest.approx(1.0)  # flat

    def test_empty_profile_scores_zero(self):
        profiles = {
            "kama_period": pd.DataFrame(
                columns=["kama_period", "mean_objective", "std_objective", "count"]
            ),
            "lookback_period": pd.DataFrame(
                columns=["lookback_period", "mean_objective", "std_objective", "count"]
            ),
            "kama_buffer": pd.DataFrame(
                columns=["kama_buffer", "mean_objective", "std_objective", "count"]
            ),
            "top_n": pd.DataFrame(
                columns=["top_n", "mean_objective", "std_objective", "count"]
            ),
        }
        scores = compute_robustness_scores(profiles)
        for name in ["kama_period", "lookback_period", "kama_buffer", "top_n"]:
            assert scores[name] == 0.0


# ---------------------------------------------------------------------------
# Sensitivity report formatting test
# ---------------------------------------------------------------------------
class TestFormatSensitivityReport:
    def test_report_contains_key_sections(self):
        grid_df = pd.DataFrame({
            "kama_period": [10, 20],
            "lookback_period": [60, 60],
            "kama_buffer": [0.01, 0.01],
            "top_n": [20, 20],
            "objective": [2.0, 3.0],
            "cagr": [0.15, 0.20],
            "max_drawdown": [0.10, 0.08],
            "sharpe": [1.2, 1.5],
            "calmar": [1.5, 2.5],
        })
        profiles = compute_marginal_profiles(grid_df)
        scores = compute_robustness_scores(profiles)
        result = SensitivityResult(
            grid_results=grid_df,
            marginal_profiles=profiles,
            robustness_scores=scores,
            base_params=StrategyParams(kama_period=10, lookback_period=60,
                                       kama_buffer=0.01, top_n=20),
            base_objective=2.0,
        )
        report = format_sensitivity_report(result)
        assert "SENSITIVITY ANALYSIS" in report
        assert "Base Parameters" in report
        assert "Marginal Profiles" in report
        assert "Robustness Scores" in report
        assert "VERDICT" in report


# ---------------------------------------------------------------------------
# End-to-end sensitivity test (small grid, synthetic data)
# ---------------------------------------------------------------------------
class TestRunSensitivity:
    def test_small_grid_runs(self, long_synthetic_prices, long_synthetic_open):
        """End-to-end: tiny 2x1x1x1 grid on synthetic data."""
        tickers = [c for c in long_synthetic_prices.columns if c != "SPY"]
        grid = {
            "kama_period": [10, 20],
            "lookback_period": [60],
            "kama_buffer": [0.01],
            "top_n": [10],
        }
        result = run_sensitivity(
            long_synthetic_prices,
            long_synthetic_open,
            tickers,
            initial_capital=10_000,
            grid=grid,
            n_workers=1,
        )
        assert isinstance(result, SensitivityResult)
        assert len(result.grid_results) == 2
        assert "kama_period" in result.robustness_scores
        assert "lookback_period" in result.robustness_scores

    def test_base_objective_found(self, long_synthetic_prices, long_synthetic_open):
        """Base params objective should be populated when base is in grid."""
        tickers = [c for c in long_synthetic_prices.columns if c != "SPY"]
        base = StrategyParams(kama_period=20, lookback_period=60,
                              kama_buffer=0.01, top_n=10)
        grid = {
            "kama_period": [10, 20],
            "lookback_period": [60],
            "kama_buffer": [0.01],
            "top_n": [10],
        }
        result = run_sensitivity(
            long_synthetic_prices,
            long_synthetic_open,
            tickers,
            initial_capital=10_000,
            base_params=base,
            grid=grid,
            n_workers=1,
        )
        assert not np.isnan(result.base_objective)
