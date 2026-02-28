"""V2 Walk-Forward Optimization — vol-targeted KAMA momentum.

Reuses schedule generation and equity-curve stitching from the base
walk_forward module.  Swaps in the V2 engine, optimizer, and params.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import structlog

from src.portfolio_sim.config import INITIAL_CAPITAL
from src.portfolio_sim.models import WFOResult, WFOStep
from src.portfolio_sim.optimizer import (
    _get_kama_periods_from_space,
    precompute_kama_caches,
)
from src.portfolio_sim.reporting import compute_metrics
from src.portfolio_sim.walk_forward import (
    _stitch_equity_curves,
    generate_wfo_schedule,
)

from src.portfolio_sim.strategy_v2.config import (
    V2_MAX_DD_LIMIT,
    V2_SEARCH_SPACE,
)
from src.portfolio_sim.strategy_v2.engine import run_simulation_v2
from src.portfolio_sim.strategy_v2.optimizer import (
    find_best_params_v2,
    run_sensitivity_v2,
)
from src.portfolio_sim.strategy_v2.parallel import init_eval_worker_v2
from src.portfolio_sim.strategy_v2.params import StrategyParamsV2

log = structlog.get_logger()


def run_walk_forward_v2(
    close_prices: pd.DataFrame,
    open_prices: pd.DataFrame,
    tickers: list[str],
    initial_capital: float = INITIAL_CAPITAL,
    base_params: StrategyParamsV2 | None = None,
    space: dict[str, dict] | None = None,
    n_trials_per_step: int = 100,
    n_workers: int | None = None,
    min_is_days: int | None = None,
    oos_days: int | None = None,
    max_dd_limit: float = V2_MAX_DD_LIMIT,
    kama_caches: dict[int, dict[str, pd.Series]] | None = None,
    executor: ProcessPoolExecutor | None = None,
) -> WFOResult:
    """Run sliding walk-forward optimisation with V2 vol-targeted engine."""
    base_params = base_params or StrategyParamsV2()
    space = space or V2_SEARCH_SPACE
    if min_is_days is None:
        min_is_days = 126
    if oos_days is None:
        oos_days = 21
    if not n_workers or n_workers < 1:
        n_workers = os.cpu_count()

    dates = close_prices.index
    schedule = generate_wfo_schedule(dates, min_is_days, oos_days)

    if not schedule:
        raise ValueError(
            f"Not enough data for V2 WFO: {len(dates)} trading days available, "
            f"need at least {min_is_days + oos_days} "
            f"(min_is_days={min_is_days} + oos_days={oos_days})."
        )

    log.info(
        "v2_wfo_start",
        n_steps=len(schedule),
        min_is_days=min_is_days,
        oos_days=oos_days,
        n_trials_per_step=n_trials_per_step,
    )

    # Pre-compute KAMA once on full data
    if kama_caches is None:
        kama_periods = _get_kama_periods_from_space(space)
        kama_caches = precompute_kama_caches(
            close_prices, tickers, kama_periods, n_workers,
        )
        log.info("v2_wfo_kama_precomputed", n_periods=len(kama_caches))

    # Create persistent executor
    own_executor = executor is None
    if own_executor:
        executor = ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=init_eval_worker_v2,
            initargs=(close_prices, open_prices, tickers, initial_capital, kama_caches),
        )

    steps: list[WFOStep] = []

    try:
        for step_idx, (is_start, is_end, oos_start, oos_end) in enumerate(schedule):
            log.info(
                "v2_wfo_step_start",
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

            # --- 2. Optimise on IS ---
            max_warmup = base_params.warmup
            available_equity_days = max(1, len(close_is) - max_warmup)
            min_n_days_is = max(5, min(available_equity_days, 60))

            sens_result = run_sensitivity_v2(
                close_is, open_is, valid_is,
                initial_capital,
                base_params=base_params,
                space=space,
                n_trials=n_trials_per_step,
                n_workers=n_workers,
                max_dd_limit=max_dd_limit,
                min_n_days=min_n_days_is,
                kama_caches=kama_caches,
                executor=executor,
            )

            best_params = find_best_params_v2(sens_result)
            if best_params is None:
                log.warning("v2_wfo_step_no_valid_params", step=step_idx + 1)
                best_params = base_params

            # IS metrics
            is_kama_cache = kama_caches.get(best_params.kama_period)
            is_sim = run_simulation_v2(
                close_is, open_is, valid_is, initial_capital,
                params=best_params,
                kama_cache=is_kama_cache,
            )
            is_metrics = compute_metrics(is_sim.equity)

            # --- 3. Run OOS simulation with extended warmup ---
            # Extra warmup for vol-targeting estimator calibration
            warmup = best_params.warmup + best_params.portfolio_vol_lookback
            oos_start_loc = dates.get_loc(oos_start)
            warmup_start_idx = max(0, oos_start_loc - warmup)
            oos_end_loc = dates.get_loc(oos_end)

            close_oos_warm = close_prices.iloc[warmup_start_idx:oos_end_loc + 1]
            open_oos_warm = open_prices.iloc[warmup_start_idx:oos_end_loc + 1]

            valid_oos = [
                t for t in tickers
                if t in close_oos_warm.columns and len(close_oos_warm[t].dropna()) >= 5
            ]

            oos_kama_cache = kama_caches.get(best_params.kama_period)
            oos_sim = run_simulation_v2(
                close_oos_warm, open_oos_warm, valid_oos, initial_capital,
                params=best_params,
                kama_cache=oos_kama_cache,
            )

            # Trim to OOS period only
            oos_equity = oos_sim.equity.loc[oos_start:oos_end]
            oos_spy_equity = oos_sim.spy_equity.loc[oos_start:oos_end]

            if oos_equity.empty:
                log.warning("v2_wfo_step_empty_oos", step=step_idx + 1)
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
                "v2_wfo_step_done",
                step=step_idx + 1,
                is_sharpe=f"{is_metrics.get('sharpe', 0):.2f}",
                oos_sharpe=f"{oos_metrics.get('sharpe', 0):.2f}",
            )
    finally:
        if own_executor:
            executor.shutdown(wait=True)

    if not steps:
        raise ValueError("V2 WFO produced no valid steps.")

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
        "v2_wfo_done",
        n_steps=len(steps),
        oos_sharpe=f"{oos_metrics.get('sharpe', 0):.2f}",
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
# Report formatting
# ---------------------------------------------------------------------------
def format_wfo_report_v2(result: WFOResult) -> str:
    """Format a human-readable V2 walk-forward optimisation report."""
    lines: list[str] = []
    lines.append("=" * 90)
    lines.append("WALK-FORWARD OPTIMIZATION REPORT (V2: Vol-Targeted)")
    lines.append("=" * 90)

    lines.append("")
    lines.append(f"Schedule: Sliding WFO, {len(result.steps)} steps")
    if result.steps:
        s0 = result.steps[0]
        lines.append(f"  Full period: {s0.is_start.date()} .. {result.steps[-1].oos_end.date()}")

    # Per-step table
    lines.append("")
    lines.append("-" * 90)
    header = (
        f"  {'Step':>4}  {'IS Period':<25} {'OOS Period':<25}"
        f"  {'IS Sharpe':>9}  {'OOS Sharpe':>10}  {'OOS MaxDD':>9}"
    )
    lines.append(header)
    lines.append("-" * 90)

    is_sharpes = []
    oos_sharpes = []
    for step in result.steps:
        is_sharpe = step.is_metrics.get("sharpe", 0.0)
        oos_sharpe = step.oos_metrics.get("sharpe", 0.0)
        oos_maxdd = step.oos_metrics.get("max_drawdown", 0.0)
        is_sharpes.append(is_sharpe)
        oos_sharpes.append(oos_sharpe)

        is_range = f"{step.is_start.date()}..{step.is_end.date()}"
        oos_range = f"{step.oos_start.date()}..{step.oos_end.date()}"

        lines.append(
            f"  {step.step_index + 1:>4}  {is_range:<25} {oos_range:<25}"
            f"  {is_sharpe:>9.2f}  {oos_sharpe:>10.2f}  {oos_maxdd:>9.2%}"
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
    lines.append(f"  Ann. Vol:       {om.get('annualized_vol', 0):.2%}")
    lines.append(f"  Trading Days:   {om.get('n_days', 0)}")

    # IS/OOS degradation (Sharpe-based)
    if is_sharpes and oos_sharpes:
        avg_is = np.mean(is_sharpes)
        avg_oos = np.mean(oos_sharpes)
        if avg_is > 0:
            degradation = 1.0 - avg_oos / avg_is
        else:
            degradation = float("nan")

        lines.append("")
        lines.append("-" * 90)
        lines.append("IS/OOS Degradation Analysis (Sharpe):")
        lines.append("-" * 90)
        lines.append(f"  Average IS Sharpe:   {avg_is:>8.2f}")
        lines.append(f"  Average OOS Sharpe:  {avg_oos:>8.2f}")
        lines.append(f"  Degradation:         {degradation:>8.1%}")
        if degradation <= 0.5:
            lines.append("  Verdict: ACCEPTABLE (< 50% degradation)")
        else:
            lines.append("  Verdict: HIGH DEGRADATION — possible overfitting")

    # Recommended live parameters (V2-specific)
    fp = result.final_params
    lines.append("")
    lines.append("-" * 90)
    lines.append("Recommended Live Parameters (from final IS window):")
    lines.append("-" * 90)
    lines.append(f"  kama_period:           {fp.kama_period}")
    lines.append(f"  lookback_period:       {fp.lookback_period}")
    lines.append(f"  kama_buffer:           {fp.kama_buffer}")
    lines.append(f"  top_n:                 {fp.top_n}")
    lines.append(f"  oos_days:              {fp.oos_days}")
    lines.append(f"  corr_threshold:        {fp.corr_threshold}")
    lines.append(f"  weighting_mode:        {fp.weighting_mode}")
    # V2-specific params
    lines.append(f"  target_vol:            {fp.target_vol}")
    lines.append(f"  max_leverage:          {fp.max_leverage}")
    lines.append(f"  portfolio_vol_lookback: {fp.portfolio_vol_lookback}")

    lines.append("")
    lines.append("=" * 90)
    return "\n".join(lines)
