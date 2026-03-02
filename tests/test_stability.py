"""Tests for parameter stability analysis module."""

import numpy as np
import pandas as pd
import pytest

from src.portfolio_sim.config import SEARCH_SPACE
from src.portfolio_sim.stability import (
    ParamStabilityPoint,
    ParamStabilityResult,
    StabilityAnalysisResult,
    _compute_cv,
    _verdict,
    format_stability_report,
    generate_candidate_values,
)
from src.portfolio_sim.params import StrategyParams


# ---------------------------------------------------------------------------
# generate_candidate_values
# ---------------------------------------------------------------------------

class TestGenerateCandidateValues:
    def test_categorical_returns_choices(self):
        values = generate_candidate_values("r2_window", SEARCH_SPACE)
        assert values == [20, 25, 28, 30, 35, 45, 60]

    def test_categorical_top_n(self):
        values = generate_candidate_values("top_n", SEARCH_SPACE)
        assert values == [5, 8, 10, 12, 15]

    def test_int_generates_range(self):
        values = generate_candidate_values("portfolio_vol_lookback", SEARCH_SPACE)
        assert values == [15, 20, 25, 30, 35]

    def test_float_generates_arange(self):
        values = generate_candidate_values("kama_buffer", SEARCH_SPACE)
        assert len(values) == 7
        assert values[0] == pytest.approx(0.02)
        assert values[-1] == pytest.approx(0.05)

    def test_float_target_vol(self):
        values = generate_candidate_values("target_vol", SEARCH_SPACE)
        assert len(values) == 7
        assert values[0] == pytest.approx(0.08)
        assert values[-1] == pytest.approx(0.20)

    def test_float_min_invested_pct(self):
        values = generate_candidate_values("min_invested_pct", SEARCH_SPACE)
        assert len(values) == 10
        assert values[0] == pytest.approx(0.0)
        assert values[-1] == pytest.approx(0.9)

    def test_max_points_subsampling(self):
        values = generate_candidate_values("r2_window", SEARCH_SPACE, max_points=3)
        assert len(values) == 3
        assert values[0] == 20   # first
        assert values[-1] == 60  # last

    def test_max_points_none_returns_all(self):
        values = generate_candidate_values("r2_window", SEARCH_SPACE, max_points=None)
        assert len(values) == 7

    def test_max_points_equal_to_count(self):
        values = generate_candidate_values("max_leverage", SEARCH_SPACE, max_points=4)
        assert len(values) == 4

    def test_max_points_greater_than_count(self):
        values = generate_candidate_values("max_leverage", SEARCH_SPACE, max_points=10)
        assert len(values) == 4

    def test_unknown_type_raises(self):
        bad_space = {"x": {"type": "unknown", "choices": [1]}}
        with pytest.raises(ValueError, match="Unknown param type"):
            generate_candidate_values("x", bad_space)


# ---------------------------------------------------------------------------
# CV and verdict helpers
# ---------------------------------------------------------------------------

class TestCVAndVerdict:
    def test_compute_cv_normal(self):
        cv = _compute_cv([10.0, 10.0, 10.0])
        assert cv == pytest.approx(0.0)

    def test_compute_cv_varied(self):
        cv = _compute_cv([1.0, 2.0, 3.0])
        assert cv > 0

    def test_compute_cv_insufficient_data(self):
        assert np.isnan(_compute_cv([1.0]))
        assert np.isnan(_compute_cv([]))

    def test_compute_cv_zero_mean(self):
        assert np.isnan(_compute_cv([-1.0, 1.0]))

    def test_verdict_stable(self):
        assert _verdict(0.05) == "STABLE"

    def test_verdict_moderate(self):
        assert _verdict(0.15) == "MODERATE"

    def test_verdict_fragile(self):
        assert _verdict(0.30) == "FRAGILE"

    def test_verdict_nan(self):
        assert _verdict(float("nan")) == "N/A"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

def _make_point(name, value, sharpe=1.0, cagr=0.1, failed=False):
    metrics = {"sharpe": sharpe, "cagr": cagr, "max_drawdown": 0.15, "calmar": 0.8,
               "total_return": 0.2, "annualized_vol": 0.12} if not failed else {}
    return ParamStabilityPoint(
        param_name=name, param_value=value,
        oos_metrics=metrics, n_wfo_steps=5, failed=failed,
    )


class TestParamStabilityResult:
    def test_to_dataframe(self):
        pr = ParamStabilityResult(
            param_name="top_n", base_value=10,
            points=[_make_point("top_n", 5), _make_point("top_n", 10), _make_point("top_n", 15)],
        )
        df = pr.to_dataframe()
        assert len(df) == 3
        assert "param_value" in df.columns
        assert "oos_sharpe" in df.columns
        assert "oos_cagr" in df.columns
        assert "failed" in df.columns

    def test_to_dataframe_with_failed(self):
        pr = ParamStabilityResult(
            param_name="top_n", base_value=10,
            points=[_make_point("top_n", 5), _make_point("top_n", 10, failed=True)],
        )
        df = pr.to_dataframe()
        assert len(df) == 2
        assert df.iloc[1]["failed"] == True  # noqa: E712 (numpy bool)
        assert np.isnan(df.iloc[1]["oos_sharpe"])


class TestStabilityAnalysisResult:
    def test_summary_dataframe(self):
        result = StabilityAnalysisResult(
            param_results={
                "top_n": ParamStabilityResult(
                    param_name="top_n", base_value=10,
                    points=[_make_point("top_n", 5), _make_point("top_n", 10)],
                ),
                "r2_window": ParamStabilityResult(
                    param_name="r2_window", base_value=30,
                    points=[_make_point("r2_window", 20), _make_point("r2_window", 30)],
                ),
            },
            base_params=StrategyParams(),
        )
        df = result.summary_dataframe()
        assert len(df) == 4
        assert set(df["param_name"].unique()) == {"top_n", "r2_window"}
        assert "oos_sharpe" in df.columns


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

class TestFormatReport:
    def test_report_contains_all_params(self):
        result = StabilityAnalysisResult(
            param_results={
                "top_n": ParamStabilityResult(
                    param_name="top_n", base_value=10,
                    points=[
                        _make_point("top_n", 5, sharpe=0.9),
                        _make_point("top_n", 10, sharpe=1.0),
                        _make_point("top_n", 15, sharpe=0.95),
                    ],
                ),
            },
            base_params=StrategyParams(),
        )
        report = format_stability_report(result, metric="sharpe")
        assert "PARAMETER STABILITY ANALYSIS" in report
        assert "top_n" in report
        assert "base=10" in report
        assert "SUMMARY" in report
        assert "Verdict" in report
