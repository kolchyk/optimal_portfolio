"""Walk-Forward Optimization (WFO) for KAMA momentum strategy.

Splits the timeline into expanding in-sample (IS) windows for parameter
optimization and fixed out-of-sample (OOS) windows for validation.
This prevents overfitting by never testing parameters on data they were
trained on.

Anchored WFO: IS always starts at the first available date and grows
by *oos_days* each step.  The final step's optimized parameters are
the recommended "live" parameters.
"""

from __future__ import annotations

import os

import pandas as pd
import structlog

from src.portfolio_sim.config import INITIAL_CAPITAL
from src.portfolio_sim.engine import run_simulation
from src.portfolio_sim.models import WFOResult, WFOStep
from src.portfolio_sim.optimizer import (
    SENSITIVITY_SPACE,
    _get_kama_periods_from_space,
    compute_objective,
    find_best_params,
    precompute_kama_caches,
    run_sensitivity,
)
from src.portfolio_sim.params import StrategyParams
from src.portfolio_sim.reporting import compute_metrics

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Schedule generation
# ---------------------------------------------------------------------------
def generate_wfo_schedule(
    dates: pd.DatetimeIndex,
    min_is_days: int = 756,
    oos_days: int = 252,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Generate (is_start, is_end, oos_start, oos_end) tuples.

    Anchored WFO: IS always starts at ``dates[0]``, IS end advances by
    *oos_days* each step.

    Returns empty list if there is not enough data for even one step.
    """
    total = len(dates)
    schedule: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    is_end_idx = min_is_days - 1

    while is_end_idx + oos_days < total:
        is_start = dates[0]
        is_end = dates[is_end_idx]
        oos_start = dates[is_end_idx + 1]
        oos_end_idx = min(is_end_idx + oos_days, total - 1)
        oos_end = dates[oos_end_idx]

        schedule.append((is_start, is_end, oos_start, oos_end))
        is_end_idx += oos_days

    return schedule


# ---------------------------------------------------------------------------
# Main walk-forward optimization
# ---------------------------------------------------------------------------
def run_walk_forward(
    close_prices: pd.DataFrame,
    open_prices: pd.DataFrame,
    tickers: list[str],
    initial_capital: float = INITIAL_CAPITAL,
    base_params: StrategyParams | None = None,
    space: dict[str, dict] | None = None,
    n_trials_per_step: int = 50,
    n_workers: int | None = None,
    min_is_days: int = 756,
    oos_days: int = 252,
    max_dd_limit: float = 0.30,
) -> WFOResult:
    """Run anchored walk-forward optimization.

    For each step:
      1. Optimize parameters on the IS (in-sample) data slice.
      2. Run simulation with optimized params on the OOS (out-of-sample)
         data slice (with warmup prefix for indicator computation).
      3. Record IS and OOS metrics.

    After all steps the OOS equity curves are stitched together
    to produce the aggregate out-of-sample performance.
    """
    base_params = base_params or StrategyParams()
    space = space or SENSITIVITY_SPACE
    n_workers = n_workers or max(1, os.cpu_count() - 1)

    dates = close_prices.index
    schedule = generate_wfo_schedule(dates, min_is_days, oos_days)

    if not schedule:
        raise ValueError(
            f"Not enough data for WFO: {len(dates)} trading days available, "
            f"need at least {min_is_days + oos_days} "
            f"(min_is_days={min_is_days} + oos_days={oos_days})."
        )

    log.info(
        "wfo_start",
        n_steps=len(schedule),
        min_is_days=min_is_days,
        oos_days=oos_days,
        n_trials_per_step=n_trials_per_step,
    )

    steps: list[WFOStep] = []

    for step_idx, (is_start, is_end, oos_start, oos_end) in enumerate(schedule):
        log.info(
            "wfo_step_start",
            step=step_idx + 1,
            is_range=f"{is_start.date()}..{is_end.date()}",
            oos_range=f"{oos_start.date()}..{oos_end.date()}",
        )

        # --- 1. Slice IS data ---
        close_is = close_prices.loc[is_start:is_end]
        open_is = open_prices.loc[is_start:is_end]
        valid_is = [
            t for t in tickers
            if t in close_is.columns and len(close_is[t].dropna()) >= min_is_days // 2
        ]

        # --- 2. Optimize on IS ---
        sens_result = run_sensitivity(
            close_is, open_is, valid_is,
            initial_capital,
            base_params=base_params,
            space=space,
            n_trials=n_trials_per_step,
            n_workers=n_workers,
            max_dd_limit=max_dd_limit,
        )

        best_params = find_best_params(sens_result)
        if best_params is None:
            log.warning("wfo_step_no_valid_params", step=step_idx + 1)
            best_params = base_params

        # Run IS simulation with best params for IS metrics
        is_sim = run_simulation(
            close_is, open_is, valid_is, initial_capital,
            params=best_params,
        )
        is_metrics = compute_metrics(is_sim.equity)

        # --- 3. Run OOS simulation with warmup prefix ---
        warmup = best_params.warmup
        oos_start_loc = dates.get_loc(oos_start)
        warmup_start_idx = max(0, oos_start_loc - warmup)
        oos_end_loc = dates.get_loc(oos_end)

        close_oos_warm = close_prices.iloc[warmup_start_idx:oos_end_loc + 1]
        open_oos_warm = open_prices.iloc[warmup_start_idx:oos_end_loc + 1]

        valid_oos = [
            t for t in tickers
            if t in close_oos_warm.columns and len(close_oos_warm[t].dropna()) >= warmup
        ]

        oos_sim = run_simulation(
            close_oos_warm, open_oos_warm, valid_oos, initial_capital,
            params=best_params,
        )

        # Trim equity to only the OOS period
        oos_equity = oos_sim.equity.loc[oos_start:oos_end]
        oos_spy_equity = oos_sim.spy_equity.loc[oos_start:oos_end]

        if oos_equity.empty:
            log.warning("wfo_step_empty_oos", step=step_idx + 1)
            continue

        oos_metrics = compute_metrics(oos_equity)

        step = WFOStep(
            step_index=step_idx,
            is_start=is_start,
            is_end=is_end,
            oos_start=oos_start,
            oos_end=oos_end,
            optimized_params=best_params,
            is_metrics=is_metrics,
            oos_metrics=oos_metrics,
            oos_equity=oos_equity,
            oos_spy_equity=oos_spy_equity,
        )
        steps.append(step)

        log.info(
            "wfo_step_done",
            step=step_idx + 1,
            is_cagr=f"{is_metrics.get('cagr', 0):.2%}",
            oos_cagr=f"{oos_metrics.get('cagr', 0):.2%}",
        )

    if not steps:
        raise ValueError("WFO produced no valid steps.")

    # --- 4. Stitch OOS equity curves ---
    stitched_equity = _stitch_equity_curves(
        [s.oos_equity for s in steps], initial_capital,
    )
    stitched_spy = _stitch_equity_curves(
        [s.oos_spy_equity for s in steps], initial_capital,
    )

    oos_metrics = compute_metrics(stitched_equity)
    final_params = steps[-1].optimized_params

    log.info(
        "wfo_done",
        n_steps=len(steps),
        oos_cagr=f"{oos_metrics.get('cagr', 0):.2%}",
        oos_maxdd=f"{oos_metrics.get('max_drawdown', 0):.2%}",
    )

    return WFOResult(
        steps=steps,
        stitched_equity=stitched_equity,
        stitched_spy_equity=stitched_spy,
        oos_metrics=oos_metrics,
        final_params=final_params,
    )


# ---------------------------------------------------------------------------
# Equity curve stitching
# ---------------------------------------------------------------------------
def _stitch_equity_curves(
    curves: list[pd.Series],
    initial_capital: float,
) -> pd.Series:
    """Concatenate OOS equity curves, scaling each segment.

    Each segment is scaled so it starts where the previous segment ended.
    The first segment is scaled to start at *initial_capital*.
    """
    segments: list[pd.Series] = []
    current_value = initial_capital

    for curve in curves:
        if curve.empty:
            continue
        scale = current_value / curve.iloc[0]
        scaled = curve * scale
        segments.append(scaled)
        current_value = scaled.iloc[-1]

    if not segments:
        return pd.Series(dtype=float)

    return pd.concat(segments)


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------
def format_wfo_report(result: WFOResult) -> str:
    """Format a human-readable walk-forward optimization report."""
    lines: list[str] = []
    lines.append("=" * 90)
    lines.append("WALK-FORWARD OPTIMIZATION REPORT")
    lines.append("=" * 90)

    lines.append("")
    lines.append(f"Schedule: Anchored WFO, {len(result.steps)} steps")
    if result.steps:
        s0 = result.steps[0]
        lines.append(f"  Full period: {s0.is_start.date()} .. {result.steps[-1].oos_end.date()}")

    # Per-step table
    lines.append("")
    lines.append("-" * 90)
    header = (
        f"  {'Step':>4}  {'IS Period':<25} {'OOS Period':<25}"
        f"  {'IS CAGR':>8}  {'OOS CAGR':>9}  {'OOS MaxDD':>9}"
    )
    lines.append(header)
    lines.append("-" * 90)

    is_cagrs = []
    oos_cagrs = []
    for step in result.steps:
        is_cagr = step.is_metrics.get("cagr", 0.0)
        oos_cagr = step.oos_metrics.get("cagr", 0.0)
        oos_maxdd = step.oos_metrics.get("max_drawdown", 0.0)
        is_cagrs.append(is_cagr)
        oos_cagrs.append(oos_cagr)

        is_range = f"{step.is_start.date()}..{step.is_end.date()}"
        oos_range = f"{step.oos_start.date()}..{step.oos_end.date()}"

        lines.append(
            f"  {step.step_index + 1:>4}  {is_range:<25} {oos_range:<25}"
            f"  {is_cagr:>8.2%}  {oos_cagr:>9.2%}  {oos_maxdd:>9.2%}"
        )

    # Stitched OOS performance
    lines.append("")
    lines.append("-" * 90)
    lines.append("Stitched Out-of-Sample Performance:")
    lines.append("-" * 90)

    om = result.oos_metrics
    lines.append(f"  CAGR:           {om.get('cagr', 0):.2%}")
    lines.append(f"  Total Return:   {om.get('total_return', 0):.2%}")
    lines.append(f"  Max Drawdown:   {om.get('max_drawdown', 0):.2%}")
    lines.append(f"  Sharpe:         {om.get('sharpe', 0):.2f}")
    lines.append(f"  Calmar:         {om.get('calmar', 0):.2f}")
    lines.append(f"  Trading Days:   {om.get('n_days', 0)}")

    # IS/OOS degradation
    if is_cagrs and oos_cagrs:
        import numpy as np
        avg_is = np.mean(is_cagrs)
        avg_oos = np.mean(oos_cagrs)
        if avg_is > 0:
            degradation = 1.0 - avg_oos / avg_is
        else:
            degradation = float("nan")

        lines.append("")
        lines.append("-" * 90)
        lines.append("IS/OOS Degradation Analysis:")
        lines.append("-" * 90)
        lines.append(f"  Average IS CAGR:   {avg_is:>8.2%}")
        lines.append(f"  Average OOS CAGR:  {avg_oos:>8.2%}")
        lines.append(f"  Degradation:       {degradation:>8.1%}")
        if degradation <= 0.5:
            lines.append("  Verdict: ACCEPTABLE (< 50% degradation)")
        else:
            lines.append("  Verdict: HIGH DEGRADATION — possible overfitting")

    # Recommended live parameters
    fp = result.final_params
    lines.append("")
    lines.append("-" * 90)
    lines.append("Recommended Live Parameters (from final IS window):")
    lines.append("-" * 90)
    lines.append(f"  kama_period:     {fp.kama_period}")
    lines.append(f"  lookback_period: {fp.lookback_period}")
    lines.append(f"  kama_buffer:     {fp.kama_buffer}")
    lines.append(f"  top_n:           {fp.top_n}")

    lines.append("")
    lines.append("=" * 90)
    return "\n".join(lines)
