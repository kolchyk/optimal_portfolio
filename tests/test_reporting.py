"""Tests for reporting module."""

import numpy as np
import pandas as pd
import pytest

from src.portfolio_sim.reporting import (
    compute_drawdown_series,
    compute_metrics,
    format_comparison_table,
    format_metrics_table,
    compute_monthly_returns,
    compute_yearly_returns,
    compute_rolling_sharpe,
    save_equity_png,
    format_asset_report,
)
from src.portfolio_sim.models import SimulationResult


def test_compute_monthly_returns():
    """Verify monthly returns table structure."""
    # Start earlier to ensure Jan is covered after pct_change
    dates = pd.date_range("2022-12-01", periods=150, freq="D")
    eq = pd.Series(np.linspace(100, 110, 150), index=dates)
    table = compute_monthly_returns(eq)
    assert isinstance(table, pd.DataFrame)
    assert "Jan" in table.columns
    assert 2023 in table.index


def test_compute_yearly_returns():
    """Verify yearly returns series."""
    dates = pd.date_range("2022-01-01", periods=500, freq="D")
    eq = pd.Series(np.linspace(100, 120, 500), index=dates)
    yr = compute_yearly_returns(eq)
    assert isinstance(yr, pd.Series)
    assert 2023 in yr.index


def test_compute_rolling_sharpe():
    """Verify rolling Sharpe ratio."""
    dates = pd.date_range("2023-01-01", periods=300, freq="B")
    eq = pd.Series(np.exp(np.linspace(0, 0.1, 300)), index=dates)
    rs = compute_rolling_sharpe(eq, window=20)
    assert len(rs) == 300 - 1 - 20 + 1  # diff drops 1, rolling window drops 19


def test_save_equity_png(tmp_path):
    """Test saving PNG (with mock plt)."""
    dates = pd.date_range("2023-01-01", periods=10)
    eq = pd.Series(np.linspace(100, 110, 10), index=dates)
    spy = pd.Series(np.linspace(100, 105, 10), index=dates)
    
    path = save_equity_png(eq, spy, tmp_path)
    assert path.exists()
    assert path.name == "equity_curve.png"


def test_format_asset_report():
    """Verify asset report formatting."""
    dates = pd.date_range("2023-01-01", periods=5)
    eq = pd.Series([10000, 10100, 10200, 10300, 10400], index=dates)
    holdings = pd.DataFrame(
        {"AAPL": [0, 10, 10, 10, 0], "MSFT": [0, 0, 5, 5, 5]},
        index=dates
    )
    cash = pd.Series([10000, 8500, 7000, 7000, 8600], index=dates)
    trade_log = [
        {"date": "2023-01-02", "ticker": "AAPL", "action": "buy", "shares": 10, "price": 150},
        {"date": "2023-01-05", "ticker": "AAPL", "action": "sell", "shares": 10, "price": 160},
    ]
    
    sim_result = SimulationResult(
        equity=eq,
        spy_equity=eq * 0.95,
        holdings_history=holdings,
        cash_history=cash,
        trade_log=trade_log,
        regime_history=None
    )
    
    close_prices = pd.DataFrame(
        {"AAPL": [150, 152, 155, 158, 160], "MSFT": [250, 252, 255, 258, 260]},
        index=dates
    )
    
    report = format_asset_report(sim_result, close_prices)
    assert "ASSET REPORT" in report
    assert "AAPL" in report
    assert "MSFT" in report
    assert "Unique assets traded: 2" in report


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
