"""Tests for reporting module."""

import numpy as np
import pandas as pd
import pytest

from src.portfolio_sim.reporting import (
    compute_drawdown_series,
    compute_metrics,
    format_metrics_table,
    save_wfv_report,
    save_wfv_json,
)


def test_metrics_constant_equity():
    """Constant equity → zero returns and zero drawdown."""
    eq = pd.Series([100.0] * 252)
    m = compute_metrics(eq)
    assert m["total_return"] == pytest.approx(0.0, abs=1e-6)
    assert m["max_drawdown"] == pytest.approx(0.0, abs=1e-6)
    assert m["n_days"] == 252


def test_metrics_linear_growth():
    """Linearly growing equity."""
    eq = pd.Series(np.linspace(100, 200, 252))
    m = compute_metrics(eq)
    assert m["total_return"] == pytest.approx(1.0, abs=0.01)  # 100% return
    assert m["cagr"] > 0
    assert m["max_drawdown"] == pytest.approx(0.0, abs=1e-6)  # No drawdown for monotonic increase


def test_metrics_with_drawdown():
    """Equity that goes up then down."""
    eq = pd.Series([100, 120, 110, 115, 105])
    m = compute_metrics(eq)
    assert m["total_return"] == pytest.approx(0.05, abs=0.01)
    assert m["max_drawdown"] > 0  # Should detect the drawdown


def test_metrics_empty_equity():
    eq = pd.Series([], dtype=float)
    m = compute_metrics(eq)
    assert m["total_return"] == 0.0
    assert m["n_days"] == 0


def test_drawdown_series_shape():
    eq = pd.Series([100, 110, 105, 120, 115])
    dd = compute_drawdown_series(eq)
    assert len(dd) == len(eq)
    assert dd.iloc[0] == 0.0  # First value: no drawdown
    assert dd.max() <= 0.0 + 1e-10  # All values should be <= 0


def test_drawdown_series_values():
    eq = pd.Series([100, 120, 100, 120])
    dd = compute_drawdown_series(eq)
    # After peak of 120, drop to 100 → drawdown = (100-120)/120 = -1/6
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


def test_save_wfv_report(tmp_path):
    oos_equity = pd.Series(
        [10000, 10500, 10200, 10800],
        index=pd.date_range("2023-01-01", periods=4),
    )
    wfv_result = {
        "oos_equity": oos_equity,
        "windows": [
            {
                "window": 1,
                "train_start": "2022-01-01",
                "train_end": "2022-12-31",
                "test_start": "2023-01-01",
                "test_end": "2023-04-01",
                "params": {"kama_period": 20, "lookback_period": 126, "max_correlation": 0.7, "top_n_selection": 15},
                "is_score": 1.5,
                "oos_return_pct": 8.0,
                "oos_max_dd_pct": -2.9,
            }
        ],
    }
    path = save_wfv_report(wfv_result, metric="calmar", output_dir=tmp_path)
    assert path.exists()
    content = path.read_text()
    assert "Walk-Forward Validation Report" in content
    assert "calmar" in content.lower()


def test_save_wfv_json(tmp_path):
    wfv_result = {
        "windows": [
            {
                "window": 1,
                "params": {"kama_period": 20},
                "is_score": 1.0,
                "oos_return_pct": 5.0,
                "oos_max_dd_pct": -3.0,
            }
        ],
    }
    path = save_wfv_json(wfv_result, output_dir=tmp_path)
    assert path.exists()
    import json
    data = json.loads(path.read_text())
    assert len(data) == 1
    assert data[0]["window"] == 1
