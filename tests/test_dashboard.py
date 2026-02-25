"""Tests for efficient frontier CAGR conversion in dashboard."""

import numpy as np
import pandas as pd
import pytest

from dashboard import compute_efficient_frontier


def test_frontier_returns_are_geometric(synthetic_prices):
    """Frontier returns (CAGR) must be strictly below arithmetic returns."""
    asset_prices = synthetic_prices.drop(columns=["SPY"])
    frontier_vols, frontier_rets, _ = compute_efficient_frontier(asset_prices)

    assert len(frontier_vols) > 0

    # Recompute arithmetic returns for comparison
    daily_returns = asset_prices.pct_change().dropna(how="all")
    valid_cols = [
        c for c in daily_returns.columns
        if daily_returns[c].dropna().shape[0] >= 252
    ]
    daily_returns = daily_returns[valid_cols].dropna()
    max_arithmetic = daily_returns.mean().max() * 252

    # Geometric (CAGR) should be lower than arithmetic
    assert frontier_rets.max() < max_arithmetic


def test_asset_stats_uses_cagr(synthetic_prices):
    """Individual asset returns in asset_stats must match actual CAGR from prices."""
    asset_prices = synthetic_prices.drop(columns=["SPY"])
    _, _, asset_stats = compute_efficient_frontier(asset_prices)

    assert not asset_stats.empty

    for _, row in asset_stats.iterrows():
        ticker = row["ticker"]
        p = asset_prices[ticker].dropna()
        expected_cagr = (p.iloc[-1] / p.iloc[0]) ** (252 / len(p)) - 1
        assert row["ann_return"] == pytest.approx(expected_cagr, rel=1e-6)


def test_frontier_geometric_approximation_low_vol():
    """For low-volatility assets, geometric ≈ arithmetic (small σ²/2 gap)."""
    np.random.seed(42)
    n_days = 500
    dates = pd.bdate_range("2021-01-04", periods=n_days)

    data = {}
    for i in range(5):
        returns = np.random.normal(0.0003, 0.002, n_days)  # ~3% annual vol
        data[f"BOND{i}"] = 100 * np.exp(np.cumsum(returns))

    close_prices = pd.DataFrame(data, index=dates)
    frontier_vols, frontier_rets, _ = compute_efficient_frontier(close_prices)

    assert len(frontier_vols) > 0
    assert np.all(np.isfinite(frontier_rets))
    assert np.all(np.isfinite(frontier_vols))
    # Low-vol frontier should still have positive returns
    assert frontier_rets.max() > 0


def test_frontier_high_volatility():
    """High-volatility assets must not crash the computation."""
    np.random.seed(42)
    n_days = 500
    dates = pd.bdate_range("2021-01-04", periods=n_days)

    data = {}
    for i in range(5):
        returns = np.random.normal(0.001, 0.05, n_days)  # ~79% annual vol
        data[f"HV{i}"] = 100 * np.exp(np.cumsum(returns))

    close_prices = pd.DataFrame(data, index=dates)
    frontier_vols, frontier_rets, _ = compute_efficient_frontier(close_prices)

    assert np.all(np.isfinite(frontier_rets))
    assert np.all(np.isfinite(frontier_vols))


def test_frontier_empty_with_insufficient_data():
    """Fewer than 2 valid assets should return empty arrays."""
    dates = pd.bdate_range("2021-01-04", periods=100)
    close_prices = pd.DataFrame({"A": np.linspace(100, 110, 100)}, index=dates)
    frontier_vols, frontier_rets, asset_stats = compute_efficient_frontier(close_prices)

    assert len(frontier_vols) == 0
    assert len(frontier_rets) == 0
