"""Walk-Forward Optimization for Clenow R2 Momentum Strategy.

Optimizes R2 Momentum parameters (r2_lookback, KAMA periods, gap filter,
ATR sizing, etc.) using sliding IS/OOS windows with Optuna TPE sampler.
Objective: maximize CAGR (with MaxDD < 30% guard).

Usage:
    uv run python scripts/wfo_r2_momentum.py
    uv run python scripts/wfo_r2_momentum.py --n-trials 20 --period 3y
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, fields
from functools import partial
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from tqdm import tqdm

from scripts.compare_methods import run_backtest
from src.portfolio_sim.cli_utils import (
    create_output_dir,
    filter_valid_tickers,
    setup_logging,
)
from src.portfolio_sim.config import (
    INITIAL_CAPITAL,
    KAMA_BUFFER,
    KAMA_PERIOD,
    KAMA_SPY_PERIOD,
    SPY_TICKER,
    TOP_N,
)
from src.portfolio_sim.data import fetch_etf_tickers, fetch_price_data
from src.portfolio_sim.optimizer import (
    _get_kama_periods_from_space,
    precompute_kama_caches,
)
from src.portfolio_sim.reporting import compute_metrics, save_equity_png
from src.portfolio_sim.walk_forward import (
    _stitch_equity_curves,
    generate_wfo_schedule,
)


# ---------------------------------------------------------------------------
# R2 Strategy Parameters
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class R2StrategyParams:
    """Immutable parameter set for R2 Momentum backtest."""

    r2_lookback: int = 90
    kama_asset_period: int = KAMA_PERIOD
    kama_spy_period: int = KAMA_SPY_PERIOD
    kama_buffer: float = KAMA_BUFFER
    gap_threshold: float = 0.15
    atr_period: int = 20
    risk_factor: float = 0.001
    top_n: int = TOP_N
    rebal_period_weeks: int = 2

    @property
    def warmup(self) -> int:
        return max(self.r2_lookback, self.kama_asset_period, self.kama_spy_period) + 10


R2_PARAM_NAMES = [f.name for f in fields(R2StrategyParams)]

R2_SEARCH_SPACE: dict[str, dict] = {
    "r2_lookback": {"type": "int", "low": 20, "high": 120, "step": 20},
    "kama_asset_period": {"type": "categorical", "choices": [10, 20, 30, 40, 50]},
    "kama_spy_period": {"type": "categorical", "choices": [20, 30, 40, 50]},
    "kama_buffer": {"type": "float", "low": 0.005, "high": 0.03, "step": 0.005},
    "gap_threshold": {"type": "float", "low": 0.10, "high": 0.20, "step": 0.025},
    "atr_period": {"type": "int", "low": 10, "high": 30, "step": 5},
    "top_n": {"type": "int", "low": 5, "high": 25, "step": 5},
    "rebal_period_weeks": {"type": "int", "low": 1, "high": 6, "step": 1},
}

_METRIC_KEYS = ["cagr", "max_drawdown", "sharpe", "calmar"]


# ---------------------------------------------------------------------------
# Objective: maximize CAGR
# ---------------------------------------------------------------------------
def r2_objective(
    equity: pd.Series,
    max_dd_limit: float = 0.30,
    min_n_days: int = 20,
) -> float:
    """CAGR-maximizing objective with MaxDD guard."""
    metrics = compute_metrics(equity)
    if metrics["n_days"] < min_n_days:
        return -999.0
    if metrics["max_drawdown"] > max_dd_limit:
        return -999.0
    cagr = metrics["cagr"]
    if cagr <= 0:
        return -999.0
    return cagr


# ---------------------------------------------------------------------------
# Worker infrastructure
# ---------------------------------------------------------------------------
_r2_shared: dict = {}


def _init_r2_worker(
    close_prices: pd.DataFrame,
    open_prices: pd.DataFrame,
    tickers: list[str],
    initial_capital: float,
    kama_caches: dict[int, dict[str, pd.Series]],
) -> None:
    """Initializer for R2 evaluation worker processes."""
    _r2_shared["close"] = close_prices
    _r2_shared["open"] = open_prices
    _r2_shared["tickers"] = tickers
    _r2_shared["capital"] = initial_capital
    _r2_shared["kama_caches"] = kama_caches


def _evaluate_r2_combo(args: tuple) -> dict:
    """Evaluate a single R2 param combo in a worker process."""
    params, max_dd_limit, min_n_days, slice_spec = args

    close = _r2_shared["close"]
    open_ = _r2_shared["open"]
    tickers = _r2_shared["tickers"]
    capital = _r2_shared["capital"]
    kama_caches = _r2_shared["kama_caches"]

    # Slice data for IS window
    if slice_spec is not None:
        sim_start = slice_spec.get("sim_start")
        sim_end = slice_spec.get("sim_end")
        if sim_start:
            close = close.loc[sim_start:]
            open_ = open_.loc[sim_start:]
        if sim_end:
            close = close.loc[:sim_end]
            open_ = open_.loc[:sim_end]
        tickers = slice_spec.get("tickers", tickers)

    # Resolve KAMA caches for this param combo
    asset_kama = kama_caches.get(params.kama_asset_period, {})
    spy_kama_cache = kama_caches.get(params.kama_spy_period, {})
    spy_kama = spy_kama_cache.get(SPY_TICKER)

    try:
        equity, _, _ = run_backtest(
            close, open_, tickers,
            initial_capital=capital,
            top_n=params.top_n,
            rebal_period_weeks=params.rebal_period_weeks,
            kama_asset_period=params.kama_asset_period,
            kama_spy_period=params.kama_spy_period,
            kama_buffer=params.kama_buffer,
            r2_lookback=params.r2_lookback,
            gap_threshold=params.gap_threshold,
            atr_period=params.atr_period,
            risk_factor=params.risk_factor,
            kama_cache_ext=asset_kama,
            spy_kama_ext=spy_kama,
        )

        if equity.empty:
            obj = -999.0
            metrics = {}
        else:
            obj = r2_objective(equity, max_dd_limit, min_n_days)
            metrics = compute_metrics(equity)
    except (ValueError, KeyError):
        obj = -999.0
        metrics = {}

    row: dict = {}
    for key in R2_PARAM_NAMES:
        row[key] = getattr(params, key)
    row["objective"] = obj
    for key in _METRIC_KEYS:
        row[key] = metrics.get(key, 0.0)
    return row


def _suggest_r2_params(
    trial: optuna.Trial,
    space: dict[str, dict],
) -> R2StrategyParams:
    """Suggest R2StrategyParams from an Optuna trial."""
    kwargs = {}
    for name, spec in space.items():
        if spec["type"] == "categorical":
            kwargs[name] = trial.suggest_categorical(name, spec["choices"])
        elif spec["type"] == "int":
            kwargs[name] = trial.suggest_int(
                name, spec["low"], spec["high"], step=spec.get("step", 1),
            )
        elif spec["type"] == "float":
            kwargs[name] = trial.suggest_float(
                name, spec["low"], spec["high"], step=spec.get("step"),
            )
    return R2StrategyParams(**kwargs)


# ---------------------------------------------------------------------------
# Optuna batch loop
# ---------------------------------------------------------------------------
def _run_r2_optuna_loop(
    study: optuna.Study,
    executor: ProcessPoolExecutor,
    *,
    space: dict[str, dict],
    n_trials: int,
    n_workers: int,
    max_dd_limit: float,
    min_n_days: int,
    slice_spec: dict | None = None,
    desc: str = "R2 trials",
) -> pd.DataFrame:
    """Batch-parallel Optuna ask/tell loop for R2 strategy."""
    pbar = tqdm(total=n_trials, desc=desc, unit="trial")
    trials_done = 0

    while trials_done < n_trials:
        batch_size = min(n_workers, n_trials - trials_done)
        trials = [study.ask() for _ in range(batch_size)]
        params_list = [_suggest_r2_params(t, space) for t in trials]

        futures = {
            executor.submit(
                _evaluate_r2_combo,
                (p, max_dd_limit, min_n_days, slice_spec),
            ): (t, p)
            for t, p in zip(trials, params_list)
        }
        for future in as_completed(futures):
            trial, _ = futures[future]
            result_dict = future.result()
            obj = result_dict["objective"]
            value = obj if obj > -999.0 else float("-inf")
            for key in _METRIC_KEYS:
                trial.set_user_attr(key, result_dict.get(key, 0.0))
            study.tell(trial, value)
            pbar.update(1)
            trials_done += 1

    pbar.close()

    rows: list[dict] = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            row = dict(trial.params)
            obj_val = trial.value
            row["objective"] = obj_val if obj_val != float("-inf") else -999.0
            row.update(trial.user_attrs)
            rows.append(row)

    return pd.DataFrame(rows)


def _select_best_r2_params(grid_df: pd.DataFrame) -> R2StrategyParams | None:
    """Select best R2StrategyParams from optimization grid."""
    valid = grid_df[grid_df["objective"] > -999.0]
    if valid.empty:
        return None
    best = valid.loc[valid["objective"].idxmax()]

    _field_types = {f.name: f.type for f in fields(R2StrategyParams)}
    kwargs = {}
    for key in R2_PARAM_NAMES:
        if key in best.index:
            val = best[key]
            expected = _field_types.get(key)
            if expected == "int" or expected is int:
                val = int(val)
            elif expected == "float" or expected is float:
                val = float(val)
            kwargs[key] = val
    return R2StrategyParams(**kwargs)


# ---------------------------------------------------------------------------
# Walk-Forward loop
# ---------------------------------------------------------------------------
@dataclass
class R2WFOStep:
    """One step of R2 walk-forward optimization."""

    step_index: int
    is_start: pd.Timestamp
    is_end: pd.Timestamp
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp
    optimized_params: R2StrategyParams
    is_metrics: dict
    oos_metrics: dict
    oos_equity: pd.Series


@dataclass
class R2WFOResult:
    """Complete R2 walk-forward result."""

    steps: list[R2WFOStep]
    stitched_equity: pd.Series
    stitched_spy_equity: pd.Series
    oos_metrics: dict
    final_params: R2StrategyParams


def _get_r2_kama_periods(space: dict[str, dict]) -> list[int]:
    """Extract KAMA periods from R2 search space."""
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
    max_dd_limit: float = 0.30,
) -> R2WFOResult:
    """Run sliding walk-forward optimization for R2 Momentum strategy."""
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

    print(f"  WFO schedule: {len(schedule)} steps")
    print(f"  IS window: {min_is_days} days, OOS window: {oos_days} days")
    print(f"  Trials per step: {n_trials_per_step}, Workers: {n_workers}")

    # Pre-compute KAMA on full data
    kama_periods = _get_r2_kama_periods(space)
    kama_caches = precompute_kama_caches(close_prices, tickers, kama_periods, n_workers)

    # Persistent executor for all steps
    executor = ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_r2_worker,
        initargs=(close_prices, open_prices, tickers, initial_capital, kama_caches),
    )

    base_params = R2StrategyParams()
    steps: list[R2WFOStep] = []

    try:
        for step_idx, (is_start, is_end, oos_start, oos_end) in enumerate(schedule):
            print(f"\n  Step {step_idx + 1}/{len(schedule)}: "
                  f"IS {is_start.date()}..{is_end.date()} | "
                  f"OOS {oos_start.date()}..{oos_end.date()}")

            # --- 1. Optimize on IS ---
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=42 + step_idx),
            )

            # Compute min_n_days for IS (warmup eats bars)
            max_warmup = base_params.warmup
            close_is = close_prices.loc[is_start:is_end]
            available_days = max(1, len(close_is) - max_warmup)
            min_n_days_is = max(5, min(available_days, 60))

            # Shared executor has full data; use slice_spec
            slice_spec = {
                "sim_start": is_start,
                "sim_end": is_end,
                "tickers": tickers,
            }

            grid_df = _run_r2_optuna_loop(
                study, executor,
                space=space,
                n_trials=n_trials_per_step,
                n_workers=n_workers,
                max_dd_limit=max_dd_limit,
                min_n_days=min_n_days_is,
                slice_spec=slice_spec,
                desc=f"Step {step_idx + 1}",
            )

            best_params = _select_best_r2_params(grid_df)
            if best_params is None:
                print(f"    WARNING: No valid params found, using defaults")
                best_params = base_params

            # IS metrics with best params
            asset_kama = kama_caches.get(best_params.kama_asset_period, {})
            spy_kama_cache = kama_caches.get(best_params.kama_spy_period, {})
            spy_kama = spy_kama_cache.get(SPY_TICKER)

            is_equity, _, _ = run_backtest(
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

            # --- 2. Run OOS with warmup prefix ---
            warmup = best_params.warmup
            oos_start_loc = dates.get_loc(oos_start)
            warmup_start_idx = max(0, oos_start_loc - warmup)
            oos_end_loc = dates.get_loc(oos_end)

            close_oos = close_prices.iloc[warmup_start_idx:oos_end_loc + 1]
            open_oos = open_prices.iloc[warmup_start_idx:oos_end_loc + 1]

            oos_equity_full, _, _ = run_backtest(
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

            # Trim to OOS period only
            oos_equity = oos_equity_full.loc[oos_start:oos_end]
            if oos_equity.empty:
                print(f"    WARNING: Empty OOS equity, skipping step")
                continue

            oos_metrics = compute_metrics(oos_equity)

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

    # --- 3. Stitch OOS equity ---
    stitched_equity = _stitch_equity_curves(
        [s.oos_equity for s in steps], initial_capital,
    )

    # SPY benchmark for stitched period
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
    """Format a human-readable R2 WFO report."""
    lines: list[str] = []
    lines.append("=" * 90)
    lines.append("R2 MOMENTUM — WALK-FORWARD OPTIMIZATION REPORT")
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

    # Recommended live parameters
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
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Walk-Forward Optimization for R2 Momentum Strategy",
    )
    parser.add_argument("--period", default="3y", help="yfinance period (default: 3y)")
    parser.add_argument("--n-trials", type=int, default=50, help="Optuna trials per step (default: 50)")
    parser.add_argument("--n-workers", type=int, default=None, help="Parallel workers (default: cpu_count - 1)")
    parser.add_argument("--min-is-days", type=int, default=126, help="IS window days (default: 126)")
    parser.add_argument("--oos-days", type=int, default=21, help="OOS window days (default: 21)")
    parser.add_argument("--refresh", action="store_true", help="Force refresh data cache")
    args = parser.parse_args()

    setup_logging()

    print("=" * 60)
    print("R2 Momentum — Walk-Forward Optimization")
    print("=" * 60)

    # 1. Load data
    print("\nLoading data...")
    tickers = fetch_etf_tickers()
    close_prices, open_prices = fetch_price_data(
        tickers, period=args.period, refresh=args.refresh, cache_suffix="_etf",
    )

    min_days = 100  # R2 lookback can be up to 120
    valid_tickers = filter_valid_tickers(close_prices, min_days)
    print(f"  Universe: {len(valid_tickers)} tickers with {min_days}+ days")
    print(f"  Date range: {close_prices.index[0].date()} to {close_prices.index[-1].date()}")

    if not valid_tickers:
        print(f"\nERROR: No tickers with {min_days}+ trading days.")
        return

    # 2. Run WFO
    print("\nRunning walk-forward optimization...")
    result = run_r2_walk_forward(
        close_prices, open_prices, valid_tickers,
        initial_capital=INITIAL_CAPITAL,
        n_trials_per_step=args.n_trials,
        n_workers=args.n_workers,
        min_is_days=args.min_is_days,
        oos_days=args.oos_days,
    )

    # 3. Report
    report = format_r2_wfo_report(result)
    print(f"\n{report}")

    # 4. Save artifacts
    output_dir = create_output_dir("wfo_r2")
    print(f"\nSaving artifacts to {output_dir}/")

    (output_dir / "wfo_report.txt").write_text(report, encoding="utf-8")
    result.stitched_equity.to_csv(output_dir / "stitched_oos_equity.csv", header=True)

    save_equity_png(
        result.stitched_equity,
        result.stitched_spy_equity,
        output_dir,
        title="R2 Momentum — Walk-Forward OOS Equity (Stitched)",
        filename="stitched_oos_equity.png",
    )

    # Step details CSV
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


if __name__ == "__main__":
    main()
