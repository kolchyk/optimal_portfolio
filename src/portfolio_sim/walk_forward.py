"""Walk-Forward Optimization (WFO) for hybrid R² Momentum + vol-targeting strategy.

Splits the timeline into fixed-size in-sample (IS) windows for parameter
optimization and fixed out-of-sample (OOS) windows for validation.

Sliding WFO: both IS start and IS end advance by *oos_days* each step,
keeping the IS window at a constant *min_is_days* width.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import optuna
import pandas as pd
import structlog

from src.portfolio_sim.config import (
    INITIAL_CAPITAL,
    MAX_DD_LIMIT,
    SCHEDULE_SEARCH_SPACE,
    SEARCH_SPACE,
    get_kama_periods,
)
from src.portfolio_sim.engine import run_simulation
from src.portfolio_sim.models import WFOResult, WFOStep
from src.portfolio_sim.optimizer import (
    find_best_params,
    precompute_kama_caches,
    run_sensitivity,
)
from src.portfolio_sim.parallel import init_eval_worker
from src.portfolio_sim.params import StrategyParams
from src.portfolio_sim.reporting import compute_metrics

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Schedule generation
# ---------------------------------------------------------------------------
def generate_wfo_schedule(
    dates: pd.DatetimeIndex,
    min_is_days: int = 90,
    oos_days: int = 90,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Generate (is_start, is_end, oos_start, oos_end) tuples.

    Sliding WFO: IS window is always *min_is_days* wide and advances by
    *oos_days* each step.
    """
    if oos_days > min_is_days:
        raise ValueError(
            f"OOS period ({oos_days}d) must be \u2264 IS period ({min_is_days}d)"
        )

    total = len(dates)
    schedule: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    is_end_idx = min_is_days - 1

    while is_end_idx + oos_days < total:
        is_start = dates[is_end_idx - min_is_days + 1]
        is_end = dates[is_end_idx]
        oos_start = dates[is_end_idx + 1]
        oos_end_idx = min(is_end_idx + oos_days, total - 1)
        oos_end = dates[oos_end_idx]

        schedule.append((is_start, is_end, oos_start, oos_end))
        is_end_idx += oos_days

    return schedule


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
# Walk-Forward Optimization
# ---------------------------------------------------------------------------
def run_walk_forward(
    close_prices: pd.DataFrame,
    open_prices: pd.DataFrame,
    tickers: list[str],
    initial_capital: float = INITIAL_CAPITAL,
    base_params: StrategyParams | None = None,
    space: dict[str, dict] | None = None,
    n_trials_per_step: int = 100,
    n_workers: int | None = None,
    min_is_days: int | None = None,
    oos_days: int | None = None,
    max_dd_limit: float = MAX_DD_LIMIT,
    metric: str = "total_return",
    kama_caches: dict[int, dict[str, pd.Series]] | None = None,
    executor: ProcessPoolExecutor | None = None,
    verbose: bool = True,
    high_prices: pd.DataFrame | None = None,
    low_prices: pd.DataFrame | None = None,
) -> WFOResult:
    """Run sliding walk-forward optimisation."""
    base_params = base_params or StrategyParams()
    space = space or SEARCH_SPACE
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

    # Pre-compute KAMA once on full data
    if kama_caches is None:
        kama_periods = get_kama_periods(space)
        kama_caches = precompute_kama_caches(
            close_prices, tickers, kama_periods, n_workers,
        )
        log.info("wfo_kama_precomputed", n_periods=len(kama_caches))

    # Create persistent executor
    own_executor = executor is None
    if own_executor:
        executor = ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=init_eval_worker,
            initargs=(close_prices, open_prices, tickers, initial_capital, kama_caches,
                      high_prices, low_prices),
        )

    steps: list[WFOStep] = []

    try:
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
            high_is = high_prices.loc[is_start:is_end] if high_prices is not None else None
            low_is = low_prices.loc[is_start:is_end] if low_prices is not None else None
            valid_is = [
                t for t in tickers
                if t in close_is.columns and len(close_is[t].dropna()) >= min_is_days // 2
            ]

            # --- 2. Optimise on IS ---
            max_warmup = base_params.warmup
            available_equity_days = max(1, len(close_is) - max_warmup)
            min_n_days_is = max(5, min(available_equity_days, 60))

            sens_result = run_sensitivity(
                close_is, open_is, valid_is,
                initial_capital,
                base_params=base_params,
                space=space,
                n_trials=n_trials_per_step,
                n_workers=n_workers,
                max_dd_limit=max_dd_limit,
                min_n_days=min_n_days_is,
                metric=metric,
                kama_caches=kama_caches,
                executor=executor,
                verbose=verbose,
                high_prices=high_is,
                low_prices=low_is,
            )

            best_params = find_best_params(sens_result)
            if best_params is None:
                log.warning("wfo_step_no_valid_params", step=step_idx + 1)
                best_params = base_params

            # IS metrics
            is_kama_cache = kama_caches.get(best_params.kama_asset_period)

            is_sim = run_simulation(
                close_is, open_is, valid_is, initial_capital,
                params=best_params,
                kama_cache=is_kama_cache,
                high_prices=high_is,
                low_prices=low_is,
            )
            is_metrics = compute_metrics(is_sim.equity)

            # --- 3. Run OOS simulation with extended warmup ---
            warmup = best_params.warmup + best_params.portfolio_vol_lookback
            oos_start_loc = dates.get_loc(oos_start)
            warmup_start_idx = max(0, oos_start_loc - warmup)
            oos_end_loc = dates.get_loc(oos_end)

            close_oos_warm = close_prices.iloc[warmup_start_idx:oos_end_loc + 1]
            open_oos_warm = open_prices.iloc[warmup_start_idx:oos_end_loc + 1]
            high_oos_warm = high_prices.iloc[warmup_start_idx:oos_end_loc + 1] if high_prices is not None else None
            low_oos_warm = low_prices.iloc[warmup_start_idx:oos_end_loc + 1] if low_prices is not None else None

            valid_oos = [
                t for t in tickers
                if t in close_oos_warm.columns and len(close_oos_warm[t].dropna()) >= 5
            ]

            oos_kama_cache = kama_caches.get(best_params.kama_asset_period)

            oos_sim = run_simulation(
                close_oos_warm, open_oos_warm, valid_oos, initial_capital,
                params=best_params,
                kama_cache=oos_kama_cache,
                high_prices=high_oos_warm,
                low_prices=low_oos_warm,
            )

            # Trim to OOS period only
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
                is_sharpe=f"{is_metrics.get('sharpe', 0):.2f}",
                oos_sharpe=f"{oos_metrics.get('sharpe', 0):.2f}",
            )
    finally:
        if own_executor:
            executor.shutdown(wait=True)

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
# Schedule optimization (outer Optuna loop over oos_weeks / min_is_weeks)
# ---------------------------------------------------------------------------
def run_schedule_optimization(
    close_prices: pd.DataFrame,
    open_prices: pd.DataFrame,
    tickers: list[str],
    initial_capital: float = INITIAL_CAPITAL,
    base_params: StrategyParams | None = None,
    space: dict[str, dict] | None = None,
    schedule_space: dict[str, dict] | None = None,
    n_trials_per_step: int = 100,
    n_schedule_trials: int = 20,
    n_workers: int | None = None,
    max_dd_limit: float = MAX_DD_LIMIT,
    metric: str = "total_return",
    high_prices: pd.DataFrame | None = None,
    low_prices: pd.DataFrame | None = None,
) -> WFOResult:
    """Optimize WFO schedule params via outer Optuna loop."""
    base_params = base_params or StrategyParams()
    space = space or SEARCH_SPACE
    schedule_space = schedule_space or SCHEDULE_SEARCH_SPACE
    if not n_workers or n_workers < 1:
        n_workers = os.cpu_count()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    outer_study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=123),
    )

    kama_periods = get_kama_periods(space)
    kama_caches = precompute_kama_caches(
        close_prices, tickers, kama_periods, n_workers,
    )

    executor = ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=init_eval_worker,
        initargs=(close_prices, open_prices, tickers, initial_capital, kama_caches,
                  high_prices, low_prices),
    )

    results: list[dict] = []

    try:
        for trial_idx in range(n_schedule_trials):
            trial = outer_study.ask()
            oos_weeks = trial.suggest_int(
                "oos_weeks",
                schedule_space["oos_weeks"]["low"],
                schedule_space["oos_weeks"]["high"],
                step=schedule_space["oos_weeks"]["step"],
            )
            min_is_weeks = trial.suggest_int(
                "min_is_weeks",
                schedule_space["min_is_weeks"]["low"],
                schedule_space["min_is_weeks"]["high"],
                step=schedule_space["min_is_weeks"]["step"],
            )

            oos_d = oos_weeks * 5
            min_is_d = min_is_weeks * 5

            if oos_weeks > min_is_weeks:
                outer_study.tell(trial, -999.0)
                continue

            print(f"\n  Schedule trial {trial_idx + 1}/{n_schedule_trials}: "
                  f"oos_weeks={oos_weeks} ({oos_d}d), "
                  f"min_is_weeks={min_is_weeks} ({min_is_d}d)")

            try:
                result = run_walk_forward(
                    close_prices, open_prices, tickers,
                    initial_capital=initial_capital,
                    base_params=base_params,
                    space=space,
                    n_trials_per_step=n_trials_per_step,
                    n_workers=n_workers,
                    min_is_days=min_is_d,
                    oos_days=oos_d,
                    max_dd_limit=max_dd_limit,
                    metric=metric,
                    kama_caches=kama_caches,
                    executor=executor,
                    verbose=False,
                    high_prices=high_prices,
                    low_prices=low_prices,
                )
                sharpe = result.oos_metrics.get("sharpe", -999.0)
                calmar = result.oos_metrics.get("calmar", -999.0)
                cagr = result.oos_metrics.get("cagr", 0.0)
                maxdd = result.oos_metrics.get("max_drawdown", 0.0)
                obj = sharpe if sharpe > -999.0 else -999.0
            except (ValueError, Exception) as e:
                print(f"    FAILED: {e}")
                obj = -999.0
                sharpe = calmar = cagr = maxdd = 0.0
                result = None

            outer_study.tell(trial, obj)

            results.append({
                "oos_weeks": oos_weeks,
                "min_is_weeks": min_is_weeks,
                "oos_days": oos_d,
                "min_is_days": min_is_d,
                "sharpe": sharpe,
                "calmar": calmar,
                "cagr": cagr,
                "max_drawdown": maxdd,
            })

            print(f"    CAGR={cagr:.2%}  Sharpe={sharpe:.2f}  "
                  f"MaxDD={maxdd:.2%}  Calmar={calmar:.2f}")

        # Report all results
        results_df = pd.DataFrame(results).sort_values("sharpe", ascending=False)
        print("\n" + "=" * 80)
        print("SCHEDULE OPTIMIZATION RESULTS")
        print("=" * 80)
        print(f"{'#':>3}  {'OOS_wk':>6}  {'IS_wk':>5}  "
              f"{'CAGR':>8}  {'Sharpe':>7}  {'MaxDD':>8}  {'Calmar':>7}")
        print("-" * 80)
        for i, row in enumerate(results_df.itertuples(), 1):
            print(f"{i:>3}  {row.oos_weeks:>6}  {row.min_is_weeks:>5}  "
                  f"{row.cagr:>8.2%}  {row.sharpe:>7.2f}  "
                  f"{row.max_drawdown:>8.2%}  {row.calmar:>7.2f}")
        print("-" * 80)

        # Run best combination with full verbose output
        best = results_df.iloc[0]
        best_oos_days = int(best["oos_days"])
        best_min_is_days = int(best["min_is_days"])
        print(f"\nBest: oos_weeks={int(best['oos_weeks'])}, "
              f"min_is_weeks={int(best['min_is_weeks'])}")
        print(f"\nRe-running best combination with full output...")

        best_result = run_walk_forward(
            close_prices, open_prices, tickers,
            initial_capital=initial_capital,
            base_params=base_params,
            space=space,
            n_trials_per_step=n_trials_per_step,
            n_workers=n_workers,
            min_is_days=best_min_is_days,
            oos_days=best_oos_days,
            max_dd_limit=max_dd_limit,
            metric=metric,
            kama_caches=kama_caches,
            executor=executor,
            verbose=True,
            high_prices=high_prices,
            low_prices=low_prices,
        )
    finally:
        executor.shutdown(wait=True)

    return best_result


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------
def format_wfo_report(result: WFOResult) -> str:
    """Format a human-readable walk-forward optimisation report."""
    lines: list[str] = []
    lines.append("=" * 90)
    lines.append("WALK-FORWARD OPTIMIZATION REPORT (Hybrid R\u00b2 Momentum + Vol-Targeting)")
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

    # IS/OOS degradation
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
            lines.append("  Verdict: HIGH DEGRADATION \u2014 possible overfitting")

    # Recommended live parameters
    fp = result.final_params
    lines.append("")
    lines.append("-" * 90)
    lines.append("Recommended Live Parameters (from final IS window):")
    lines.append("-" * 90)
    lines.append(f"  r2_windows:            {fp.r2_windows}")
    lines.append(f"  r2_weights:            {fp.r2_weights}")
    lines.append(f"  kama_asset_period:     {fp.kama_asset_period}")
    lines.append(f"  kama_buffer:           {fp.kama_buffer}")
    lines.append(f"  atr_period:            {fp.atr_period}")
    lines.append(f"  risk_factor:           {fp.risk_factor}")
    lines.append(f"  top_n:                 {fp.top_n}")
    lines.append(f"  rebal_period_weeks:    {fp.rebal_period_weeks}")
    lines.append(f"  gap_threshold:         {fp.gap_threshold}")
    lines.append(f"  target_vol:            {fp.target_vol}")
    lines.append(f"  max_leverage:          {fp.max_leverage}")
    lines.append(f"  portfolio_vol_lookback: {fp.portfolio_vol_lookback}")

    lines.append("")
    lines.append("=" * 90)
    return "\n".join(lines)
