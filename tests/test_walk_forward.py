"""Tests for walk-forward validation."""

import pandas as pd
import pytest

from src.portfolio_sim.walk_forward import _chain_equity_segments, generate_wfv_windows


def test_generate_windows_basic():
    windows = generate_wfv_windows(total_days=600, train_days=252, test_days=63, buffer_days=0)
    assert len(windows) > 0
    # First window
    assert windows[0] == (0, 252, 252, 315)


def test_generate_windows_with_buffer():
    windows = generate_wfv_windows(total_days=600, train_days=252, test_days=63, buffer_days=130)
    assert len(windows) > 0
    assert windows[0][0] == 130  # starts after buffer


def test_generate_windows_insufficient_data():
    windows = generate_wfv_windows(total_days=100, train_days=252, test_days=63)
    assert len(windows) == 0


def test_windows_non_overlapping():
    """Test windows slide correctly: test of window N = train start shift."""
    windows = generate_wfv_windows(total_days=1000, train_days=252, test_days=63, buffer_days=0)
    for i in range(1, len(windows)):
        # Each window's train_start should be test_days ahead of the previous
        assert windows[i][0] == windows[i - 1][0] + 63


def test_chain_equity_basic():
    seg1 = pd.Series([100, 110, 120], index=pd.date_range("2023-01-01", periods=3))
    seg2 = pd.Series([100, 90, 95], index=pd.date_range("2023-01-04", periods=3))

    chained = _chain_equity_segments([seg1, seg2], initial_capital=1000)

    assert len(chained) == 6
    # First segment scaled: starts at 1000
    assert chained.iloc[0] == pytest.approx(1000.0)
    # End of first segment: 1000 * (120/100) = 1200
    assert chained.iloc[2] == pytest.approx(1200.0)
    # Start of second segment: should equal end of first (1200)
    assert chained.iloc[3] == pytest.approx(1200.0)


def test_chain_equity_empty():
    result = _chain_equity_segments([], initial_capital=1000)
    assert len(result) == 0


def test_chain_equity_single_segment():
    seg = pd.Series([100, 150], index=pd.date_range("2023-01-01", periods=2))
    chained = _chain_equity_segments([seg], initial_capital=500)
    assert len(chained) == 2
    assert chained.iloc[0] == pytest.approx(500.0)
    assert chained.iloc[1] == pytest.approx(750.0)
