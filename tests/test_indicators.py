"""Tests for KAMA indicator."""

import numpy as np
import pandas as pd
import pytest

from src.portfolio_sim.indicators import compute_kama, compute_kama_series


def test_kama_output_shape():
    data = np.random.RandomState(42).randn(200).cumsum() + 100
    result = compute_kama(data, period=20)
    assert len(result) == len(data)


def test_kama_nan_prefix():
    """First `period` values should be NaN."""
    data = np.arange(100, dtype=float)
    period = 10
    result = compute_kama(data, period=period)
    assert np.all(np.isnan(result[:period]))
    assert not np.isnan(result[period])


def test_kama_short_series():
    """Series shorter than period should return all NaN."""
    data = np.array([1.0, 2.0, 3.0])
    result = compute_kama(data, period=10)
    assert np.all(np.isnan(result))


def test_kama_constant_input():
    """For constant input, KAMA should equal the constant after initialization."""
    data = np.full(100, 50.0)
    result = compute_kama(data, period=10)
    valid = result[~np.isnan(result)]
    np.testing.assert_allclose(valid, 50.0, atol=1e-10)


def test_kama_tracks_trend():
    """KAMA should roughly follow a strong linear trend."""
    data = np.linspace(100, 200, 200)
    result = compute_kama(data, period=10)
    # After warm-up, KAMA should be below final price but above starting
    valid = result[~np.isnan(result)]
    assert valid[-1] > 100
    assert valid[-1] < 210  # Not too far from actual


def test_kama_series_wrapper():
    """compute_kama_series should return pd.Series with correct index."""
    idx = pd.date_range("2023-01-01", periods=100)
    prices = pd.Series(np.linspace(50, 150, 100), index=idx)
    result = compute_kama_series(prices, period=10)
    assert isinstance(result, pd.Series)
    assert len(result) == 100
    assert result.index.equals(idx)


def test_kama_with_pandas_input():
    """compute_kama should accept pd.Series input."""
    data = pd.Series(np.random.RandomState(42).randn(100).cumsum() + 100)
    result = compute_kama(data, period=20)
    assert isinstance(result, np.ndarray)
    assert len(result) == 100
