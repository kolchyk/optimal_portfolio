"""Tests for max_profit module."""

import numpy as np
import optuna
import pandas as pd
import pytest
from src.portfolio_sim.config import MAX_PROFIT_SPACE
from src.portfolio_sim.max_profit import (
    MaxProfitResult,
    compute_cagr_objective,
    format_max_profit_report,
    run_max_profit_search,
)
from src.portfolio_sim.parallel import suggest_params
from src.portfolio_sim.params import StrategyParams


def test_compute_cagr_objective_valid():
    """Valid equity curve returns its CAGR."""
    dates = pd.date_range("2023-01-01", periods=100)
    # Start at 100, end at 110 (~10% return)
    equity = pd.Series(np.linspace(100, 110, 100), index=dates)
    cagr = compute_cagr_objective(equity)
    assert cagr > 0
    assert cagr == pytest.approx((1.1) ** (252 / 100) - 1, rel=1e-5)


def test_compute_cagr_objective_rejected_dd():
    """Equity with high drawdown should be rejected."""
    dates = pd.date_range("2023-01-01", periods=100)
    # 70% drawdown
    equity = pd.Series([100] * 50 + [30] * 50, index=dates)
    assert compute_cagr_objective(equity, max_dd_limit=0.6) == -999.0


def test_compute_cagr_objective_rejected_short():
    """Too short equity should be rejected."""
    dates = pd.date_range("2023-01-01", periods=3)
    equity = pd.Series(np.linspace(100, 110, 3), index=dates)
    assert compute_cagr_objective(equity) == -999.0


def test_suggest_params():
    """Verify max profit param suggestion produces valid StrategyParams."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study()
    trial = study.ask()
    params = suggest_params(trial, MAX_PROFIT_SPACE)
    assert isinstance(params, StrategyParams)


def test_suggest_params_with_fixed():
    """Verify fixed_params override works."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study()
    trial = study.ask()
    params = suggest_params(
        trial, MAX_PROFIT_SPACE, fixed_params={"top_n": 10},
    )
    assert isinstance(params, StrategyParams)
    assert params.top_n == 10


def test_run_max_profit_search_end_to_end(synthetic_prices, synthetic_open_prices, tickers_list):
    """End-to-end small Optuna search on synthetic data."""
    space = {
        "kama_period": {"type": "categorical", "choices": [10, 20]},
        "lookback_period": {"type": "int", "low": 100, "high": 100, "step": 1},
        "top_n": {"type": "int", "low": 5, "high": 5, "step": 1},
        "kama_buffer": {"type": "float", "low": 0.01, "high": 0.01, "step": 0.01},
        "use_risk_adjusted": {"type": "categorical", "choices": [True]},
        "sizing_mode": {"type": "categorical", "choices": ["equal_weight"]},
    }

    result = run_max_profit_search(
        synthetic_prices,
        synthetic_open_prices,
        tickers_list,
        initial_capital=10000,
        space=space,
        n_trials=4,
        n_workers=-1,
    )

    assert isinstance(result, MaxProfitResult)
    assert len(result.grid_results) >= 2
    assert "objective_cagr" in result.grid_results.columns
    assert "total_return" in result.grid_results.columns


def test_format_max_profit_report():
    """Verify report formatting doesn't crash and contains key sections."""
    grid_results = pd.DataFrame({
        "kama_period": [10, 20],
        "lookback_period": [60, 60],
        "top_n": [5, 5],
        "kama_buffer": [0.01, 0.01],
        "use_risk_adjusted": [True, True],
        "sizing_mode": ["equal_weight", "equal_weight"],
        "objective_cagr": [0.15, 0.20],
        "total_return": [0.5, 0.7],
        "cagr": [0.15, 0.20],
        "max_drawdown": [0.10, 0.12],
        "sharpe": [1.5, 1.8],
        "calmar": [1.5, 1.6],
        "annualized_vol": [0.10, 0.11],
        "win_rate": [0.55, 0.60],
    })

    result = MaxProfitResult(
        universe="sp500",
        grid_results=grid_results,
        default_metrics={"cagr": 0.1, "total_return": 0.3, "max_drawdown": 0.15, "sharpe": 1.0},
        default_params=StrategyParams(),
    )

    report = format_max_profit_report(result)
    assert "MAXIMUM PROFIT SEARCH" in report
    assert "sp500" in report.lower()
    assert "Top 20 Combinations" in report
    assert "Best vs Default" in report
