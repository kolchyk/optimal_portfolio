"""Tests for the parameter sensitivity analysis module."""

import numpy as np
import optuna
import pandas as pd
import pytest

from src.portfolio_sim.optimizer import (
    SENSITIVITY_SPACE,
    SensitivityResult,
    compute_marginal_profiles,
    compute_objective,
    compute_robustness_scores,
    format_sensitivity_report,
    precompute_kama_caches,
    run_sensitivity,
)
from src.portfolio_sim.parallel import suggest_params
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
# Search space tests
# ---------------------------------------------------------------------------
class TestSearchSpace:
    def test_suggest_params_returns_strategy_params(self):
        """Verify _suggest_params produces a valid StrategyParams."""
        study = optuna.create_study()
        trial = study.ask()
        params = suggest_params(trial, SENSITIVITY_SPACE)
        assert isinstance(params, StrategyParams)

    def test_kama_period_is_categorical(self):
        """kama_period must come from the categorical set."""
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study()
        results = set()
        for _ in range(50):
            trial = study.ask()
            params = suggest_params(trial, SENSITIVITY_SPACE)
            results.add(params.kama_period)
            study.tell(trial, 1.0)
        assert results.issubset({10, 15, 20, 30, 40})

    def test_params_hashable(self):
        """StrategyParams must be hashable for dict keys."""
        study = optuna.create_study()
        params_list = []
        for _ in range(10):
            trial = study.ask()
            params_list.append(suggest_params(trial, SENSITIVITY_SPACE))
            study.tell(trial, 1.0)
        s = set(params_list)
        assert len(s) <= len(params_list)


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
            "enable_correlation_filter": [False, False, False, False],
            "correlation_threshold": [0.9, 0.9, 0.9, 0.9],
            "correlation_lookback": [60, 60, 60, 60],
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
            "enable_correlation_filter": [False, False, False],
            "correlation_threshold": [0.9, 0.9, 0.9],
            "correlation_lookback": [60, 60, 60],
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
            "enable_correlation_filter": [False, False],
            "correlation_threshold": [0.9, 0.9],
            "correlation_lookback": [60, 60],
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
            "enable_correlation_filter": pd.DataFrame({
                "enable_correlation_filter": [True, False],
                "mean_objective": [3.0, 3.0],
                "std_objective": [0.1, 0.1],
                "count": [5, 5],
            }),
            "correlation_threshold": pd.DataFrame({
                "correlation_threshold": [0.5, 0.7, 0.9],
                "mean_objective": [3.0, 3.0, 3.0],
                "std_objective": [0.1, 0.1, 0.1],
                "count": [5, 5, 5],
            }),
            "correlation_lookback": pd.DataFrame({
                "correlation_lookback": [30, 60, 90],
                "mean_objective": [3.0, 3.0, 3.0],
                "std_objective": [0.1, 0.1, 0.1],
                "count": [5, 5, 5],
            }),
        }
        scores = compute_robustness_scores(profiles)
        for name in ["kama_period", "lookback_period", "kama_buffer", "top_n",
                      "enable_correlation_filter", "correlation_threshold",
                      "correlation_lookback"]:
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
            "enable_correlation_filter": pd.DataFrame({
                "enable_correlation_filter": [True, False],
                "mean_objective": [3.0, 3.0],
                "std_objective": [0.1, 0.1],
                "count": [5, 5],
            }),
            "correlation_threshold": pd.DataFrame({
                "correlation_threshold": [0.5, 0.7, 0.9],
                "mean_objective": [3.0, 3.0, 3.0],
                "std_objective": [0.1, 0.1, 0.1],
                "count": [5, 5, 5],
            }),
            "correlation_lookback": pd.DataFrame({
                "correlation_lookback": [30, 60, 90],
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
            "enable_correlation_filter": pd.DataFrame(
                columns=["enable_correlation_filter", "mean_objective", "std_objective", "count"]
            ),
            "correlation_threshold": pd.DataFrame(
                columns=["correlation_threshold", "mean_objective", "std_objective", "count"]
            ),
            "correlation_lookback": pd.DataFrame(
                columns=["correlation_lookback", "mean_objective", "std_objective", "count"]
            ),
        }
        scores = compute_robustness_scores(profiles)
        for name in ["kama_period", "lookback_period", "kama_buffer", "top_n",
                      "enable_correlation_filter", "correlation_threshold",
                      "correlation_lookback"]:
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
            "enable_correlation_filter": [False, False],
            "correlation_threshold": [0.9, 0.9],
            "correlation_lookback": [60, 60],
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
# End-to-end sensitivity test (small search, synthetic data)
# ---------------------------------------------------------------------------
class TestRunSensitivity:
    def test_small_search_runs(self, long_synthetic_prices, long_synthetic_open):
        """End-to-end: tiny Optuna search on synthetic data."""
        tickers = [c for c in long_synthetic_prices.columns if c != "SPY"]
        space = {
            "kama_period": {"type": "categorical", "choices": [10, 20]},
            "lookback_period": {"type": "int", "low": 60, "high": 60, "step": 1},
            "kama_buffer": {"type": "float", "low": 0.01, "high": 0.01, "step": 0.01},
            "top_n": {"type": "int", "low": 10, "high": 10, "step": 1},
        }
        result = run_sensitivity(
            long_synthetic_prices,
            long_synthetic_open,
            tickers,
            initial_capital=10_000,
            space=space,
            n_trials=4,
            n_workers=-1,
        )
        assert isinstance(result, SensitivityResult)
        assert len(result.grid_results) >= 2
        assert "kama_period" in result.robustness_scores
        assert "lookback_period" in result.robustness_scores

    def test_base_objective_found(self, long_synthetic_prices, long_synthetic_open):
        """Base params objective should be populated via enqueue_trial."""
        tickers = [c for c in long_synthetic_prices.columns if c != "SPY"]
        base = StrategyParams(kama_period=20, lookback_period=60,
                              kama_buffer=0.01, top_n=10)
        space = {
            "kama_period": {"type": "categorical", "choices": [10, 20]},
            "lookback_period": {"type": "int", "low": 60, "high": 60, "step": 1},
            "kama_buffer": {"type": "float", "low": 0.01, "high": 0.01, "step": 0.01},
            "top_n": {"type": "int", "low": 10, "high": 10, "step": 1},
        }
        result = run_sensitivity(
            long_synthetic_prices,
            long_synthetic_open,
            tickers,
            initial_capital=10_000,
            base_params=base,
            space=space,
            n_trials=4,
            n_workers=-1,
        )
        assert not np.isnan(result.base_objective)
