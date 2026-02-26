"""Walk-Forward Optimization (WFO) for KAMA momentum strategy.

Splits the timeline into fixed-size in-sample (IS) windows for parameter
optimization and fixed out-of-sample (OOS) windows for validation.
This prevents overfitting by never testing parameters on data they were
trained on.

Sliding WFO: both IS start and IS end advance by *oos_days* each step,
keeping the IS window at a constant *min_is_days* width.  The final
step's optimized parameters are the recommended "live" parameters.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import structlog

from src.portfolio_sim.config import INITIAL_CAPITAL, SENSITIVITY_SPACE
from src.portfolio_sim.engine import run_simulation
from src.portfolio_sim.models import WFOGridEntry, WFOGridResult, WFOResult, WFOStep
from src.portfolio_sim.optimizer import (
    _get_kama_periods_from_space,
    compute_objective,
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
    min_is_days: int = 756,
    oos_days: int = 252,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Generate (is_start, is_end, oos_start, oos_end) tuples.

    Sliding WFO: IS window is always *min_is_days* wide and advances by
    *oos_days* each step.

    Returns empty list if there is not enough data for even one step.
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
    min_is_days: int | None = None,
    oos_days: int | None = None,
    max_dd_limit: float = 0.30,
    kama_caches: dict[int, dict[str, pd.Series]] | None = None,
    executor: ProcessPoolExecutor | None = None,
) -> WFOResult:
    """Run sliding walk-forward optimization.

    For each step:
      1. Optimize parameters on the IS (in-sample) data slice.
      2. Run simulation with optimized params on the OOS (out-of-sample)
         data slice (with warmup prefix for indicator computation).
      3. Record IS and OOS metrics.

    After all steps the OOS equity curves are stitched together
    to produce the aggregate out-of-sample performance.

    When *min_is_days* / *oos_days* are ``None`` they default to
    ``base_params.lookback_period`` and ``base_params.oos_days``
    respectively (unified parameter space).

    When *kama_caches* and/or *executor* are provided, they are reused
    across all WFO steps to avoid redundant computation and pool startup.
    """
    base_params = base_params or StrategyParams()
    space = space or SENSITIVITY_SPACE
    if min_is_days is None:
        min_is_days = base_params.lookback_period
    if oos_days is None:
        oos_days = base_params.oos_days
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

    # Pre-compute KAMA once on full data for all WFO steps (skip if provided)
    if kama_caches is None:
        kama_periods = _get_kama_periods_from_space(space)
        kama_caches = precompute_kama_caches(
            close_prices, tickers, kama_periods, n_workers,
        )
        log.info("wfo_kama_precomputed", n_periods=len(kama_caches))

    # Create a persistent executor for all WFO steps (skip if provided)
    own_executor = executor is None
    if own_executor:
        executor = ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=init_eval_worker,
            initargs=(close_prices, open_prices, tickers, initial_capital, kama_caches),
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
            valid_is = [
                t for t in tickers
                if t in close_is.columns and len(close_is[t].dropna()) >= min_is_days // 2
            ]

            # --- 2. Optimize on IS ---
            # Relax the min-equity-days threshold for short IS windows:
            # warmup consumes bars before trading starts, so with short IS
            # windows the equity curve can be much shorter than 60 days.
            max_warmup = base_params.warmup  # conservative upper bound
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
                kama_caches=kama_caches,
                executor=executor,
            )

            best_params = find_best_params(sens_result)
            if best_params is None:
                log.warning("wfo_step_no_valid_params", step=step_idx + 1)
                best_params = base_params

            # Run IS simulation with best params for IS metrics
            is_kama_cache = kama_caches.get(best_params.kama_period)
            is_sim = run_simulation(
                close_is, open_is, valid_is, initial_capital,
                params=best_params,
                kama_cache=is_kama_cache,
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
                if t in close_oos_warm.columns and len(close_oos_warm[t].dropna()) >= 5
            ]

            oos_kama_cache = kama_caches.get(best_params.kama_period)
            oos_sim = run_simulation(
                close_oos_warm, open_oos_warm, valid_oos, initial_capital,
                params=best_params,
                kama_cache=oos_kama_cache,
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
    lines.append(f"  kama_period:      {fp.kama_period}")
    lines.append(f"  lookback_period:  {fp.lookback_period}")
    lines.append(f"  kama_buffer:      {fp.kama_buffer}")
    lines.append(f"  top_n:            {fp.top_n}")
    lines.append(f"  oos_days:         {fp.oos_days}")
    lines.append(f"  corr_threshold:   {fp.corr_threshold}")

    lines.append("")
    lines.append("=" * 90)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Schedule grid search
# ---------------------------------------------------------------------------
def _generate_grid(
    space: dict[str, dict],
    total_days: int | None = None,
    min_wfo_steps: int = 3,
) -> list[tuple[int, int]]:
    """Generate all (lookback_period, oos_days) combos from unified search space.

    ``lookback_period`` is used directly as ``min_is_days`` for WFO scheduling.

    If *total_days* is given, combos where ``lookback + oos`` would not leave
    room for at least *min_wfo_steps* are excluded up-front to save compute.
    """
    lbk_spec = space["lookback_period"]
    oos_spec = space["oos_days"]
    lbk_values = list(range(lbk_spec["low"], lbk_spec["high"] + 1, lbk_spec["step"]))
    oos_values = list(range(oos_spec["low"], oos_spec["high"] + 1, oos_spec["step"]))
    combos = [(lbk, oos_d) for lbk in lbk_values for oos_d in oos_values]

    if total_days is not None:
        max_window = total_days // min_wfo_steps
        combos = [(lbk, oos_d) for lbk, oos_d in combos
                  if lbk + oos_d <= max_window]

    return combos


def run_walk_forward_grid(
    close_prices: pd.DataFrame,
    open_prices: pd.DataFrame,
    tickers: list[str],
    initial_capital: float = INITIAL_CAPITAL,
    base_params: StrategyParams | None = None,
    space: dict[str, dict] | None = None,
    n_trials_per_step: int = 50,
    n_workers: int | None = None,
    max_dd_limit: float = 0.30,
) -> WFOGridResult:
    """Grid search over (lookback_period, oos_days) WFO schedule combos.

    Uses the unified ``SEARCH_SPACE`` to derive schedule parameters:
    ``lookback_period`` maps directly to ``min_is_days``.

    Runs a full WFO for each combination and picks the best by OOS Calmar.
    Pre-computes KAMA caches and creates a single persistent executor
    shared across all grid combinations and all WFO steps.
    """
    base_params = base_params or StrategyParams()
    space = space or SENSITIVITY_SPACE
    if not n_workers or n_workers < 1:
        n_workers = os.cpu_count()

    total_days = len(close_prices)
    combos = _generate_grid(space, total_days=total_days)

    log.info("wfo_grid_start", n_combos=len(combos))

    # Pre-compute KAMA once for entire grid search
    kama_periods = _get_kama_periods_from_space(space)
    kama_caches = precompute_kama_caches(
        close_prices, tickers, kama_periods, n_workers,
    )
    log.info("wfo_grid_kama_precomputed", n_periods=len(kama_caches))

    # Create a single persistent executor for all grid combos + WFO steps
    executor = ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=init_eval_worker,
        initargs=(close_prices, open_prices, tickers, initial_capital, kama_caches),
    )

    entries: list[WFOGridEntry] = []
    rows: list[dict] = []

    try:
        for idx, (lbk, oos_d) in enumerate(combos):
            log.info(
                "wfo_grid_combo",
                combo=f"{idx + 1}/{len(combos)}",
                lookback_period=lbk,
                oos_days=oos_d,
            )
            try:
                result = run_walk_forward(
                    close_prices,
                    open_prices,
                    tickers,
                    initial_capital,
                    base_params=base_params,
                    space=space,
                    n_trials_per_step=n_trials_per_step,
                    n_workers=n_workers,
                    min_is_days=lbk,
                    oos_days=oos_d,
                    max_dd_limit=max_dd_limit,
                    kama_caches=kama_caches,
                    executor=executor,
                )
            except ValueError as exc:
                log.warning(
                    "wfo_grid_combo_skip",
                    lookback_period=lbk,
                    oos_days=oos_d,
                    reason=str(exc),
                )
                rows.append({
                    "lookback_period": lbk,
                    "oos_days": oos_d,
                    "calmar": np.nan,
                    "cagr": np.nan,
                    "max_drawdown": np.nan,
                    "sharpe": np.nan,
                    "n_steps": 0,
                })
                continue

            calmar = result.oos_metrics.get("calmar", -999.0)
            entry = WFOGridEntry(
                lookback_period=lbk,
                oos_days=oos_d,
                wfo_result=result,
                oos_calmar=calmar,
            )
            entries.append(entry)
            rows.append({
                "lookback_period": lbk,
                "oos_days": oos_d,
                "calmar": calmar,
                "cagr": result.oos_metrics.get("cagr", 0.0),
                "max_drawdown": result.oos_metrics.get("max_drawdown", 0.0),
                "sharpe": result.oos_metrics.get("sharpe", 0.0),
                "n_steps": len(result.steps),
            })
    finally:
        executor.shutdown(wait=True)

    if not entries:
        raise ValueError("WFO grid search: no valid (lookback_period, oos_days) combination.")

    best = max(entries, key=lambda e: e.oos_calmar)
    summary = pd.DataFrame(rows)

    log.info(
        "wfo_grid_done",
        best_lookback_period=best.lookback_period,
        best_oos_days=best.oos_days,
        best_calmar=f"{best.oos_calmar:.4f}",
    )

    return WFOGridResult(entries=entries, best_entry=best, summary=summary)


def format_wfo_grid_report(grid_result: WFOGridResult) -> str:
    """Format a human-readable WFO schedule grid search report."""
    lines: list[str] = []
    lines.append("=" * 90)
    lines.append("WFO SCHEDULE GRID SEARCH REPORT")
    lines.append("=" * 90)

    lines.append("")
    lines.append(f"Combinations tested: {len(grid_result.summary)}")
    lines.append(f"Valid results:       {len(grid_result.entries)}")

    # Grid table
    lines.append("")
    lines.append("-" * 90)
    header = (
        f"  {'Lookback':>8}  {'OOS days':>9}  {'Steps':>5}"
        f"  {'Calmar':>8}  {'CAGR':>8}  {'MaxDD':>8}  {'Sharpe':>8}"
    )
    lines.append(header)
    lines.append("-" * 90)

    summary = grid_result.summary.sort_values("calmar", ascending=False)
    for _, row in summary.iterrows():
        calmar_s = f"{row['calmar']:.4f}" if not np.isnan(row["calmar"]) else "  SKIP"
        cagr_s = f"{row['cagr']:.2%}" if not np.isnan(row["cagr"]) else "    N/A"
        maxdd_s = f"{row['max_drawdown']:.2%}" if not np.isnan(row["max_drawdown"]) else "    N/A"
        sharpe_s = f"{row['sharpe']:.2f}" if not np.isnan(row["sharpe"]) else "    N/A"
        lines.append(
            f"  {int(row['lookback_period']):>8}  {int(row['oos_days']):>9}  {int(row['n_steps']):>5}"
            f"  {calmar_s:>8}  {cagr_s:>8}  {maxdd_s:>8}  {sharpe_s:>8}"
        )

    # Best combo
    best = grid_result.best_entry
    lines.append("")
    lines.append("-" * 90)
    lines.append("BEST SCHEDULE:")
    lines.append(f"  lookback_period = {best.lookback_period}")
    lines.append(f"  oos_days        = {best.oos_days}")
    lines.append(f"  OOS Calmar      = {best.oos_calmar:.4f}")
    lines.append("-" * 90)

    # Append full WFO report for best combo
    lines.append("")
    lines.append(format_wfo_report(best.wfo_result))

    return "\n".join(lines)
