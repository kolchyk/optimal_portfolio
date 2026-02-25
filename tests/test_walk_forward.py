"""Tests for walk-forward optimization module."""

import numpy as np
import pandas as pd
import pytest

from src.portfolio_sim.models import WFOResult, WFOStep
from src.portfolio_sim.params import StrategyParams
from src.portfolio_sim.walk_forward import (
    _stitch_equity_curves,
    format_wfo_report,
    generate_wfo_schedule,
    run_walk_forward,
)


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
# Schedule generation tests
# ---------------------------------------------------------------------------
class TestGenerateWFOSchedule:
    def test_basic_schedule(self):
        """1600 days, min_is=756, oos=252 => at least 2 steps."""
        dates = pd.bdate_range("2018-01-02", periods=1600)
        schedule = generate_wfo_schedule(dates, min_is_days=756, oos_days=252)
        assert len(schedule) >= 2

    def test_schedule_is_sliding(self):
        """IS window slides forward — start advances, size stays constant."""
        dates = pd.bdate_range("2018-01-02", periods=1600)
        min_is = 756
        schedule = generate_wfo_schedule(dates, min_is_days=min_is, oos_days=252)
        for i, (is_start, is_end, _, _) in enumerate(schedule):
            # IS window size is always min_is_days
            is_size = dates.get_loc(is_end) - dates.get_loc(is_start) + 1
            assert is_size == min_is
            # After first step, IS start should advance
            if i > 0:
                prev_start = schedule[i - 1][0]
                assert is_start > prev_start

    def test_no_overlap_is_oos(self):
        """IS end must be strictly before OOS start."""
        dates = pd.bdate_range("2018-01-02", periods=1600)
        schedule = generate_wfo_schedule(dates, min_is_days=756, oos_days=252)
        for _, is_end, oos_start, _ in schedule:
            assert is_end < oos_start

    def test_oos_windows_are_contiguous(self):
        """OOS windows should be contiguous (no gaps)."""
        dates = pd.bdate_range("2018-01-02", periods=1600)
        schedule = generate_wfo_schedule(dates, min_is_days=756, oos_days=252)
        for i in range(1, len(schedule)):
            prev_oos_end = schedule[i - 1][3]
            curr_oos_start = schedule[i][2]
            # OOS start of step i should be 1 business day after OOS end of step i-1
            gap = (curr_oos_start - prev_oos_end).days
            assert gap <= 5, f"Gap between OOS windows too large: {gap} days"

    def test_insufficient_data_returns_empty(self):
        """With too little data, schedule should be empty."""
        dates = pd.bdate_range("2018-01-02", periods=800)
        schedule = generate_wfo_schedule(dates, min_is_days=756, oos_days=252)
        assert len(schedule) == 0

    def test_is_window_constant_size(self):
        """Each step's IS window should be the same size (sliding)."""
        dates = pd.bdate_range("2018-01-02", periods=2000)
        min_is = 400
        schedule = generate_wfo_schedule(dates, min_is_days=min_is, oos_days=200)
        for is_start, is_end, _, _ in schedule:
            is_size = dates.get_loc(is_end) - dates.get_loc(is_start) + 1
            assert is_size == min_is


# ---------------------------------------------------------------------------
# Equity curve stitching tests
# ---------------------------------------------------------------------------
class TestStitchEquityCurves:
    def test_starts_at_initial_capital(self):
        curve1 = pd.Series(
            [100, 110, 120],
            index=pd.bdate_range("2020-01-01", periods=3),
        )
        stitched = _stitch_equity_curves([curve1], initial_capital=10_000)
        assert stitched.iloc[0] == pytest.approx(10_000)

    def test_continuity_between_segments(self):
        """Second segment should start where first ended."""
        curve1 = pd.Series(
            [100, 120],
            index=pd.bdate_range("2020-01-01", periods=2),
        )
        curve2 = pd.Series(
            [200, 240],
            index=pd.bdate_range("2020-06-01", periods=2),
        )
        stitched = _stitch_equity_curves([curve1, curve2], initial_capital=10_000)
        # curve1: 10000 -> 12000
        # curve2 starts at 12000 (scaled from 200), ends at 14400
        assert stitched.iloc[0] == pytest.approx(10_000)
        assert stitched.iloc[1] == pytest.approx(12_000)
        assert stitched.iloc[2] == pytest.approx(12_000)
        assert stitched.iloc[3] == pytest.approx(14_400)

    def test_empty_curves_produce_empty(self):
        stitched = _stitch_equity_curves([], initial_capital=10_000)
        assert stitched.empty

    def test_multiple_segments_preserve_returns(self):
        """Total return of stitched curve equals product of segment returns."""
        curve1 = pd.Series(
            [100, 150],  # +50%
            index=pd.bdate_range("2020-01-01", periods=2),
        )
        curve2 = pd.Series(
            [100, 80],  # -20%
            index=pd.bdate_range("2020-06-01", periods=2),
        )
        stitched = _stitch_equity_curves([curve1, curve2], initial_capital=10_000)
        total_return = stitched.iloc[-1] / stitched.iloc[0]
        expected = 1.5 * 0.8  # product of segment returns
        assert total_return == pytest.approx(expected)


# ---------------------------------------------------------------------------
# End-to-end walk-forward test (small, synthetic)
# ---------------------------------------------------------------------------
class TestRunWalkForward:
    def test_basic_wfo_runs(self, long_synthetic_prices, long_synthetic_open):
        """End-to-end WFO with minimal trials on synthetic data."""
        tickers = [c for c in long_synthetic_prices.columns if c != "SPY"]
        space = {
            "kama_period": {"type": "categorical", "choices": [10, 20]},
            "lookback_period": {"type": "int", "low": 60, "high": 60, "step": 1},
            "kama_buffer": {"type": "float", "low": 0.01, "high": 0.01, "step": 0.01},
            "top_n": {"type": "int", "low": 5, "high": 5, "step": 1},
        }
        result = run_walk_forward(
            long_synthetic_prices,
            long_synthetic_open,
            tickers,
            initial_capital=10_000,
            space=space,
            n_trials_per_step=4,
            n_workers=-1,
            min_is_days=756,
            oos_days=252,
        )
        assert isinstance(result, WFOResult)
        assert len(result.steps) >= 1
        assert not result.stitched_equity.empty
        assert not result.stitched_spy_equity.empty
        assert isinstance(result.final_params, StrategyParams)
        assert "cagr" in result.oos_metrics

    def test_insufficient_data_raises(self, long_synthetic_prices, long_synthetic_open):
        """Should raise ValueError if data is too short for even one step."""
        tickers = [c for c in long_synthetic_prices.columns if c != "SPY"]
        # Use only 500 days — not enough for min_is=756 + oos=252
        short_close = long_synthetic_prices.iloc[:500]
        short_open = long_synthetic_open.iloc[:500]
        with pytest.raises(ValueError, match="Not enough data"):
            run_walk_forward(
                short_close, short_open, tickers,
                initial_capital=10_000,
                n_trials_per_step=4,
                n_workers=-1,
                min_is_days=756,
                oos_days=252,
            )

    def test_stitched_equity_starts_at_capital(
        self, long_synthetic_prices, long_synthetic_open
    ):
        """Stitched equity curve should start at initial_capital."""
        tickers = [c for c in long_synthetic_prices.columns if c != "SPY"]
        space = {
            "kama_period": {"type": "categorical", "choices": [10]},
            "lookback_period": {"type": "int", "low": 60, "high": 60, "step": 1},
            "kama_buffer": {"type": "float", "low": 0.01, "high": 0.01, "step": 0.01},
            "top_n": {"type": "int", "low": 5, "high": 5, "step": 1},
        }
        result = run_walk_forward(
            long_synthetic_prices,
            long_synthetic_open,
            tickers,
            initial_capital=10_000,
            space=space,
            n_trials_per_step=2,
            n_workers=-1,
            min_is_days=756,
            oos_days=252,
        )
        assert result.stitched_equity.iloc[0] == pytest.approx(10_000, rel=0.01)

    def test_steps_have_oos_dates_in_range(
        self, long_synthetic_prices, long_synthetic_open
    ):
        """OOS equity dates should fall within the declared OOS range."""
        tickers = [c for c in long_synthetic_prices.columns if c != "SPY"]
        space = {
            "kama_period": {"type": "categorical", "choices": [10, 20]},
            "lookback_period": {"type": "int", "low": 60, "high": 60, "step": 1},
            "kama_buffer": {"type": "float", "low": 0.01, "high": 0.01, "step": 0.01},
            "top_n": {"type": "int", "low": 5, "high": 5, "step": 1},
        }
        result = run_walk_forward(
            long_synthetic_prices,
            long_synthetic_open,
            tickers,
            initial_capital=10_000,
            space=space,
            n_trials_per_step=4,
            n_workers=-1,
            min_is_days=756,
            oos_days=252,
        )
        for step in result.steps:
            assert step.oos_equity.index[0] >= step.oos_start
            assert step.oos_equity.index[-1] <= step.oos_end


# ---------------------------------------------------------------------------
# Report formatting test
# ---------------------------------------------------------------------------
class TestFormatWFOReport:
    def test_report_contains_key_sections(
        self, long_synthetic_prices, long_synthetic_open
    ):
        tickers = [c for c in long_synthetic_prices.columns if c != "SPY"]
        space = {
            "kama_period": {"type": "categorical", "choices": [10]},
            "lookback_period": {"type": "int", "low": 60, "high": 60, "step": 1},
            "kama_buffer": {"type": "float", "low": 0.01, "high": 0.01, "step": 0.01},
            "top_n": {"type": "int", "low": 5, "high": 5, "step": 1},
        }
        result = run_walk_forward(
            long_synthetic_prices,
            long_synthetic_open,
            tickers,
            initial_capital=10_000,
            space=space,
            n_trials_per_step=2,
            n_workers=-1,
            min_is_days=756,
            oos_days=252,
        )
        report = format_wfo_report(result)
        assert "WALK-FORWARD" in report
        assert "Out-of-Sample" in report
        assert "Degradation" in report
        assert "Recommended Live Parameters" in report
