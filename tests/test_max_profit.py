"""Tests for max_profit module."""

import numpy as np
import pandas as pd
import pytest
from src.portfolio_sim.max_profit import (
    compute_cagr_objective,
    build_full_param_grid,
    run_max_profit_search,
    format_max_profit_report,
    MaxProfitResult,
)
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
    dates = pd.date_range("2023-01-01", periods=10)
    equity = pd.Series(np.linspace(100, 110, 10), index=dates)
    assert compute_cagr_objective(equity) == -999.0


def test_build_full_param_grid():
    """Verify grid expansion with overrides."""
    grid = {
        "kama_period": [10, 20],
        "lookback_period": [60],
        "top_n": [5],
        "kama_buffer": [0.01],
        "use_risk_adjusted": [True],
        "enable_regime_filter": [True],
        "sizing_mode": ["equal_weight"],
    }
    params_list = build_full_param_grid(grid, fixed_params={"top_n": 10})
    assert len(params_list) == 2
    assert all(p.top_n == 10 for p in params_list)
    assert params_list[0].kama_period == 10
    assert params_list[1].kama_period == 20


def test_run_max_profit_search_end_to_end(synthetic_prices, synthetic_open_prices, tickers_list):
    """End-to-end small grid search on synthetic data."""
    grid = {
        "kama_period": [10, 20],
        "lookback_period": [100],
        "top_n": [5],
        "kama_buffer": [0.01],
        "use_risk_adjusted": [True],
        "enable_regime_filter": [False],
        "sizing_mode": ["equal_weight"],
    }
    
    result = run_max_profit_search(
        synthetic_prices,
        synthetic_open_prices,
        tickers_list,
        initial_capital=10000,
        grid=grid,
        n_workers=1,  # Serial for test stability
    )
    
    assert isinstance(result, MaxProfitResult)
    assert len(result.grid_results) == 2
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
        "enable_regime_filter": [True, True],
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
