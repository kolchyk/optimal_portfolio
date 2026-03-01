"""Walk-Forward Optimization (WFO) for R² Momentum strategy.

Splits the timeline into fixed-size in-sample (IS) windows for parameter
optimization and fixed out-of-sample (OOS) windows for validation.

Sliding WFO: both IS start and IS end advance by *oos_days* each step,
keeping the IS window at a constant *min_is_days* width.
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import optuna
import pandas as pd

from src.portfolio_sim.cli_utils import (
    create_output_dir,
    filter_valid_tickers,
    setup_logging,
)
from src.portfolio_sim.config import (
    INITIAL_CAPITAL,
    R2_SEARCH_SPACE,
    SCHEDULE_SEARCH_SPACE,
    SPY_TICKER,
)
from src.portfolio_sim.data import fetch_etf_tickers, fetch_price_data
from src.portfolio_sim.engine import run_backtest
from src.portfolio_sim.models import R2WFOResult, R2WFOStep
from src.portfolio_sim.optimizer import precompute_kama_caches
from src.portfolio_sim.parallel import (
    init_r2_worker,
    run_r2_optuna_loop,
    select_best_r2_params,
)
from src.portfolio_sim.params import R2StrategyParams
from src.portfolio_sim.reporting import compute_metrics, save_equity_png


# ---------------------------------------------------------------------------
# Schedule generation (shared by all strategies)
# ---------------------------------------------------------------------------
def generate_wfo_schedule(
    dates: pd.DatetimeIndex,
    min_is_days: int = 126,
    oos_days: int = 21,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Generate (is_start, is_end, oos_start, oos_end) tuples.

    Sliding WFO: IS window is always *min_is_days* wide and advances by
    *oos_days* each step.
    """
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
# Equity curve stitching (shared by all strategies)
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
# R² KAMA periods from search space
# ---------------------------------------------------------------------------
def _get_r2_kama_periods(space: dict[str, dict]) -> list[int]:
    """Extract KAMA periods from R² search space."""
    periods: list[int] = []
    for key in ("kama_asset_period", "kama_spy_period"):
        spec = space.get(key, {})
        if not spec:
            continue
        if spec.get("type") == "categorical":
            periods.extend(spec["choices"])
        else:
            periods.extend(range(
                spec.get("low", 10),
                spec.get("high", 40) + 1,
                spec.get("step", 5),
            ))
    return sorted(set(periods))


# ---------------------------------------------------------------------------
# R² Walk-Forward Optimization
# ---------------------------------------------------------------------------
def run_r2_walk_forward(
    close_prices: pd.DataFrame,
    open_prices: pd.DataFrame,
    tickers: list[str],
    initial_capital: float = INITIAL_CAPITAL,
    space: dict[str, dict] | None = None,
    n_trials_per_step: int = 50,
    n_workers: int | None = None,
    min_is_days: int = 126,
    oos_days: int = 21,
    metric: str = "total_return",
    verbose: bool = True,
) -> R2WFOResult:
    """Run sliding walk-forward optimization for R² Momentum strategy."""
    space = space or R2_SEARCH_SPACE
    if not n_workers or n_workers < 1:
        n_workers = max(1, os.cpu_count() - 1)

    dates = close_prices.index
    schedule = generate_wfo_schedule(dates, min_is_days, oos_days)

    if not schedule:
        raise ValueError(
            f"Not enough data for WFO: {len(dates)} days available, "
            f"need at least {min_is_days + oos_days}."
        )

    if verbose:
        print(f"  WFO schedule: {len(schedule)} steps")
        print(f"  IS window: {min_is_days} days, OOS window: {oos_days} days")
        print(f"  Trials per step: {n_trials_per_step}, Workers: {n_workers}")

    kama_periods = _get_r2_kama_periods(space)
    kama_caches = precompute_kama_caches(close_prices, tickers, kama_periods, n_workers)

    executor = ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=init_r2_worker,
        initargs=(close_prices, open_prices, tickers, initial_capital, kama_caches),
    )

    base_params = R2StrategyParams()
    steps: list[R2WFOStep] = []

    try:
        for step_idx, (is_start, is_end, oos_start, oos_end) in enumerate(schedule):
            if verbose:
                print(f"\n  Step {step_idx + 1}/{len(schedule)}: "
                      f"IS {is_start.date()}..{is_end.date()} | "
                      f"OOS {oos_start.date()}..{oos_end.date()}")

            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=42 + step_idx),
            )

            close_is = close_prices.loc[is_start:is_end]

            slice_spec = {
                "sim_start": is_start,
                "sim_end": is_end,
                "tickers": tickers,
            }

            grid_df = run_r2_optuna_loop(
                study, executor,
                space=space,
                n_trials=n_trials_per_step,
                n_workers=n_workers,
                metric=metric,
                slice_spec=slice_spec,
                desc=f"Step {step_idx + 1}",
                verbose=verbose,
            )

            best_params = select_best_r2_params(grid_df)
            if best_params is None:
                if verbose:
                    print(f"    WARNING: No valid params found, using defaults")
                best_params = base_params

            asset_kama = kama_caches.get(best_params.kama_asset_period, {})
            spy_kama_cache = kama_caches.get(best_params.kama_spy_period, {})
            spy_kama = spy_kama_cache.get(SPY_TICKER)

            is_equity, _, _, _, _ = run_backtest(
                close_is, open_prices.loc[is_start:is_end], tickers,
                initial_capital=initial_capital,
                top_n=best_params.top_n,
                rebal_period_weeks=best_params.rebal_period_weeks,
                kama_asset_period=best_params.kama_asset_period,
                kama_spy_period=best_params.kama_spy_period,
                kama_buffer=best_params.kama_buffer,
                r2_lookback=best_params.r2_lookback,
                gap_threshold=best_params.gap_threshold,
                atr_period=best_params.atr_period,
                risk_factor=best_params.risk_factor,
                kama_cache_ext=asset_kama,
                spy_kama_ext=spy_kama,
            )
            is_metrics = compute_metrics(is_equity) if not is_equity.empty else {}

            warmup = best_params.warmup
            oos_start_loc = dates.get_loc(oos_start)
            warmup_start_idx = max(0, oos_start_loc - warmup)
            oos_end_loc = dates.get_loc(oos_end)

            close_oos = close_prices.iloc[warmup_start_idx:oos_end_loc + 1]
            open_oos = open_prices.iloc[warmup_start_idx:oos_end_loc + 1]

            oos_equity_full, _, _, _, _ = run_backtest(
                close_oos, open_oos, tickers,
                initial_capital=initial_capital,
                top_n=best_params.top_n,
                rebal_period_weeks=best_params.rebal_period_weeks,
                kama_asset_period=best_params.kama_asset_period,
                kama_spy_period=best_params.kama_spy_period,
                kama_buffer=best_params.kama_buffer,
                r2_lookback=best_params.r2_lookback,
                gap_threshold=best_params.gap_threshold,
                atr_period=best_params.atr_period,
                risk_factor=best_params.risk_factor,
                kama_cache_ext=asset_kama,
                spy_kama_ext=spy_kama,
            )

            oos_equity = oos_equity_full.loc[oos_start:oos_end]
            if oos_equity.empty:
                if verbose:
                    print(f"    WARNING: Empty OOS equity, skipping step")
                continue

            oos_metrics = compute_metrics(oos_equity)

            if verbose:
                print(f"    Best: r2_lb={best_params.r2_lookback} "
                      f"kama_a={best_params.kama_asset_period} "
                      f"top_n={best_params.top_n} "
                      f"rebal={best_params.rebal_period_weeks}w | "
                      f"IS CAGR={is_metrics.get('cagr', 0):.1%} "
                      f"OOS CAGR={oos_metrics.get('cagr', 0):.1%}")

            steps.append(R2WFOStep(
                step_index=step_idx,
                is_start=is_start,
                is_end=is_end,
                oos_start=oos_start,
                oos_end=oos_end,
                optimized_params=best_params,
                is_metrics=is_metrics,
                oos_metrics=oos_metrics,
                oos_equity=oos_equity,
            ))
    finally:
        executor.shutdown(wait=True)

    if not steps:
        raise ValueError("WFO produced no valid steps.")

    stitched_equity = _stitch_equity_curves(
        [s.oos_equity for s in steps], initial_capital,
    )

    spy_close = close_prices[SPY_TICKER]
    stitched_dates = stitched_equity.index
    spy_at_dates = spy_close.reindex(stitched_dates).ffill()
    if not spy_at_dates.empty and spy_at_dates.iloc[0] > 0:
        stitched_spy = initial_capital * (spy_at_dates / spy_at_dates.iloc[0])
    else:
        stitched_spy = pd.Series(initial_capital, index=stitched_dates)

    oos_metrics = compute_metrics(stitched_equity)
    final_params = steps[-1].optimized_params

    return R2WFOResult(
        steps=steps,
        stitched_equity=stitched_equity,
        stitched_spy_equity=stitched_spy,
        oos_metrics=oos_metrics,
        final_params=final_params,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def format_r2_wfo_report(result: R2WFOResult) -> str:
    """Format a human-readable R² WFO report."""
    lines: list[str] = []
    lines.append("=" * 90)
    lines.append("R2 MOMENTUM — WALK-FORWARD OPTIMIZATION REPORT")
    lines.append("=" * 90)

    lines.append("")
    lines.append(f"Schedule: Sliding WFO, {len(result.steps)} steps")
    if result.steps:
        s0 = result.steps[0]
        lines.append(f"  Full period: {s0.is_start.date()} .. {result.steps[-1].oos_end.date()}")

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

    if is_cagrs and oos_cagrs:
        avg_is = np.mean(is_cagrs)
        avg_oos = np.mean(oos_cagrs)
        degradation = 1.0 - avg_oos / avg_is if avg_is > 0 else float("nan")

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

    fp = result.final_params
    lines.append("")
    lines.append("-" * 90)
    lines.append("Recommended Live Parameters (from final IS window):")
    lines.append("-" * 90)
    lines.append(f"  r2_lookback:        {fp.r2_lookback}")
    lines.append(f"  kama_asset_period:  {fp.kama_asset_period}")
    lines.append(f"  kama_spy_period:    {fp.kama_spy_period}")
    lines.append(f"  kama_buffer:        {fp.kama_buffer}")
    lines.append(f"  gap_threshold:      {fp.gap_threshold}")
    lines.append(f"  atr_period:         {fp.atr_period}")
    lines.append(f"  risk_factor:        {fp.risk_factor}")
    lines.append(f"  top_n:              {fp.top_n}")
    lines.append(f"  rebal_period_weeks: {fp.rebal_period_weeks}")

    lines.append("")
    lines.append("=" * 90)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Schedule optimization (outer Optuna loop over oos_weeks / min_is_weeks)
# ---------------------------------------------------------------------------
def run_r2_schedule_optimization(
    close_prices: pd.DataFrame,
    open_prices: pd.DataFrame,
    tickers: list[str],
    initial_capital: float = INITIAL_CAPITAL,
    space: dict[str, dict] | None = None,
    schedule_space: dict[str, dict] | None = None,
    n_trials_per_step: int = 50,
    n_schedule_trials: int = 20,
    n_workers: int | None = None,
    metric: str = "total_return",
) -> R2WFOResult:
    """Optimize WFO schedule params (oos_weeks, min_is_weeks) via outer Optuna loop.

    Each outer trial suggests schedule params, runs a full WFO, and uses the
    stitched OOS Calmar ratio as the objective.
    """
    schedule_space = schedule_space or SCHEDULE_SEARCH_SPACE

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    outer_study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=123),
    )

    results: list[dict] = []

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

        oos_days = oos_weeks * 5
        min_is_days = min_is_weeks * 5

        print(f"\n  Schedule trial {trial_idx + 1}/{n_schedule_trials}: "
              f"oos_weeks={oos_weeks} ({oos_days}d), "
              f"min_is_weeks={min_is_weeks} ({min_is_days}d)")

        try:
            result = run_r2_walk_forward(
                close_prices, open_prices, tickers,
                initial_capital=initial_capital,
                space=space,
                n_trials_per_step=n_trials_per_step,
                n_workers=n_workers,
                min_is_days=min_is_days,
                oos_days=oos_days,
                metric=metric,
                verbose=False,
            )
            calmar = result.oos_metrics.get("calmar", -999.0)
            sharpe = result.oos_metrics.get("sharpe", -999.0)
            cagr = result.oos_metrics.get("cagr", 0.0)
            maxdd = result.oos_metrics.get("max_drawdown", 0.0)
            obj = calmar if calmar > -999.0 else -999.0
        except (ValueError, Exception) as e:
            print(f"    FAILED: {e}")
            obj = -999.0
            calmar = sharpe = cagr = maxdd = 0.0
            result = None

        outer_study.tell(trial, obj)

        results.append({
            "oos_weeks": oos_weeks,
            "min_is_weeks": min_is_weeks,
            "oos_days": oos_days,
            "min_is_days": min_is_days,
            "calmar": calmar,
            "sharpe": sharpe,
            "cagr": cagr,
            "max_drawdown": maxdd,
        })

        print(f"    CAGR={cagr:.2%}  Sharpe={sharpe:.2f}  "
              f"MaxDD={maxdd:.2%}  Calmar={calmar:.2f}")

    # Report all results
    results_df = pd.DataFrame(results).sort_values("calmar", ascending=False)
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

    best_result = run_r2_walk_forward(
        close_prices, open_prices, tickers,
        initial_capital=initial_capital,
        space=space,
        n_trials_per_step=n_trials_per_step,
        n_workers=n_workers,
        min_is_days=best_min_is_days,
        oos_days=best_oos_days,
        metric=metric,
        verbose=True,
    )
    return best_result


# ---------------------------------------------------------------------------
# Main (CLI entry point)
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Walk-Forward Optimization for R² Momentum Strategy",
    )
    parser.add_argument("--period", default="3y", help="yfinance period (default: 3y)")
    parser.add_argument("--n-trials", type=int, default=50, help="Optuna trials per step")
    parser.add_argument("--n-workers", type=int, default=None, help="Parallel workers")
    parser.add_argument("--min-is-days", type=int, default=126, help="IS window days")
    parser.add_argument("--oos-days", type=int, default=21, help="OOS window days")
    parser.add_argument(
        "--metric", default="total_return",
        choices=("total_return", "cagr", "sharpe", "calmar"),
        help="Optimization metric (default: total_return)",
    )
    parser.add_argument("--refresh", action="store_true", help="Force refresh data cache")
    args = parser.parse_args()

    setup_logging()

    print("=" * 60)
    print("R² Momentum — Walk-Forward Optimization")
    print("=" * 60)

    print("\nLoading data...")
    tickers = fetch_etf_tickers()
    close_prices, open_prices = fetch_price_data(
        tickers, period=args.period, refresh=args.refresh, cache_suffix="_etf",
    )

    min_days = 100
    valid_tickers = filter_valid_tickers(close_prices, min_days)
    print(f"  Universe: {len(valid_tickers)} tickers with {min_days}+ days")
    print(f"  Date range: {close_prices.index[0].date()} to {close_prices.index[-1].date()}")

    if not valid_tickers:
        print(f"\nERROR: No tickers with {min_days}+ trading days.")
        return

    print(f"\nRunning walk-forward optimization (metric: {args.metric})...")
    result = run_r2_walk_forward(
        close_prices, open_prices, valid_tickers,
        initial_capital=INITIAL_CAPITAL,
        n_trials_per_step=args.n_trials,
        n_workers=args.n_workers,
        min_is_days=args.min_is_days,
        oos_days=args.oos_days,
        metric=args.metric,
    )

    report = format_r2_wfo_report(result)
    print(f"\n{report}")

    output_dir = create_output_dir("wfo_r2")
    print(f"\nSaving artifacts to {output_dir}/")

    (output_dir / "wfo_report.txt").write_text(report, encoding="utf-8")
    result.stitched_equity.to_csv(output_dir / "stitched_oos_equity.csv", header=True)

    save_equity_png(
        result.stitched_equity,
        result.stitched_spy_equity,
        output_dir,
        title="R² Momentum — Walk-Forward OOS Equity (Stitched)",
        filename="stitched_oos_equity.png",
    )

    step_rows = []
    for step in result.steps:
        fp = step.optimized_params
        step_rows.append({
            "step": step.step_index + 1,
            "is_start": step.is_start.date(),
            "is_end": step.is_end.date(),
            "oos_start": step.oos_start.date(),
            "oos_end": step.oos_end.date(),
            "r2_lookback": fp.r2_lookback,
            "kama_asset_period": fp.kama_asset_period,
            "kama_spy_period": fp.kama_spy_period,
            "kama_buffer": fp.kama_buffer,
            "gap_threshold": fp.gap_threshold,
            "atr_period": fp.atr_period,
            "top_n": fp.top_n,
            "rebal_period_weeks": fp.rebal_period_weeks,
            "is_cagr": step.is_metrics.get("cagr", 0),
            "is_maxdd": step.is_metrics.get("max_drawdown", 0),
            "oos_cagr": step.oos_metrics.get("cagr", 0),
            "oos_maxdd": step.oos_metrics.get("max_drawdown", 0),
            "oos_sharpe": step.oos_metrics.get("sharpe", 0),
        })
    pd.DataFrame(step_rows).to_csv(output_dir / "wfo_steps.csv", index=False)

    fp = result.final_params
    print(f"\nRecommended live parameters:")
    print(f"  r2_lookback={fp.r2_lookback}, kama_asset={fp.kama_asset_period}, "
          f"kama_spy={fp.kama_spy_period}, kama_buffer={fp.kama_buffer}, "
          f"gap={fp.gap_threshold}, atr={fp.atr_period}, "
          f"top_n={fp.top_n}, rebal={fp.rebal_period_weeks}w")
    print("\nDone.")
