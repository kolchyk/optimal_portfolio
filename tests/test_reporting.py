"""Tests for reporting module."""

import numpy as np
import pandas as pd
import pytest

from src.portfolio_sim.reporting import (
    compute_drawdown_series,
    compute_metrics,
    format_comparison_table,
    format_metrics_table,
)


def test_metrics_constant_equity():
    """Constant equity -> zero returns and zero drawdown."""
    eq = pd.Series([100.0] * 252)
    m = compute_metrics(eq)
    assert m["total_return"] == pytest.approx(0.0, abs=1e-6)
    assert m["max_drawdown"] == pytest.approx(0.0, abs=1e-6)
    assert m["n_days"] == 252


def test_metrics_linear_growth():
    """Linearly growing equity."""
    eq = pd.Series(np.linspace(100, 200, 252))
    m = compute_metrics(eq)
    assert m["total_return"] == pytest.approx(1.0, abs=0.01)
    assert m["cagr"] > 0
    assert m["max_drawdown"] == pytest.approx(0.0, abs=1e-6)


def test_metrics_with_drawdown():
    """Equity that goes up then down."""
    eq = pd.Series([100, 120, 110, 115, 105])
    m = compute_metrics(eq)
    assert m["total_return"] == pytest.approx(0.05, abs=0.01)
    assert m["max_drawdown"] > 0


def test_metrics_empty_equity():
    eq = pd.Series([], dtype=float)
    m = compute_metrics(eq)
    assert m["total_return"] == 0.0
    assert m["n_days"] == 0


def test_drawdown_series_shape():
    eq = pd.Series([100, 110, 105, 120, 115])
    dd = compute_drawdown_series(eq)
    assert len(dd) == len(eq)
    assert dd.iloc[0] == 0.0
    assert dd.max() <= 0.0 + 1e-10


def test_drawdown_series_values():
    eq = pd.Series([100, 120, 100, 120])
    dd = compute_drawdown_series(eq)
    assert dd.iloc[2] == pytest.approx(-1 / 6, abs=1e-6)


def test_format_metrics_table():
    metrics = {
        "total_return": 0.5,
        "cagr": 0.2,
        "max_drawdown": 0.15,
        "sharpe": 1.5,
        "calmar": 1.33,
        "annualized_vol": 0.18,
        "n_days": 504,
    }
    table = format_metrics_table(metrics)
    assert "CAGR" in table
    assert "Sharpe" in table
    assert "504" in table


def test_drawdown_series_empty():
    eq = pd.Series([], dtype=float)
    dd = compute_drawdown_series(eq)
    assert len(dd) == 0


def test_metrics_zero_start():
    eq = pd.Series([0.0, 100.0, 200.0])
    m = compute_metrics(eq)
    assert m["total_return"] == 0.0
    assert m["n_days"] == 0


def test_format_comparison_table():
    strat = {
        "total_return": 1.0, "cagr": 0.15, "max_drawdown": 0.20,
        "sharpe": 1.2, "calmar": 0.75, "annualized_vol": 0.18, "n_days": 2520,
    }
    spy = {
        "total_return": 0.8, "cagr": 0.10, "max_drawdown": 0.35,
        "sharpe": 0.6, "calmar": 0.29, "annualized_vol": 0.20, "n_days": 2520,
    }
    table = format_comparison_table(strat, spy)
    assert "Strategy" in table
    assert "S&P 500" in table
    assert "CAGR" in table
