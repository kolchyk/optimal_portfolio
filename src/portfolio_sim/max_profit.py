"""Maximum profit parameter search for KAMA momentum strategy.

Unlike the sensitivity analysis in optimizer.py (which checks robustness),
this module searches for the parameter combination that maximizes CAGR
over a given period using Optuna's TPE sampler. The drawdown rejection
limit is relaxed to 60%.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import optuna
import pandas as pd
import structlog
from tqdm import tqdm

from src.portfolio_sim.config import (
    DEFAULT_MAX_PROFIT_TRIALS,
    INITIAL_CAPITAL,
    MAX_PROFIT_SPACE,
)
from src.portfolio_sim.engine import run_simulation
from src.portfolio_sim.optimizer import (
    _get_kama_periods_from_space,
    precompute_kama_caches,
)
from src.portfolio_sim.parallel import suggest_params
from src.portfolio_sim.params import StrategyParams
from src.portfolio_sim.reporting import compute_metrics

log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
ALL_PARAM_NAMES: list[str] = [
    "kama_period", "lookback_period", "top_n", "kama_buffer",
    "use_risk_adjusted", "sizing_mode",
]


@dataclass
class MaxProfitResult:
    """Results from a max-profit parameter search."""

    universe: str
    grid_results: pd.DataFrame
    default_metrics: dict
    default_params: StrategyParams


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------
def compute_cagr_objective(
    equity: pd.Series, max_dd_limit: float = 0.60,
) -> float:
    """CAGR-maximizing objective with relaxed drawdown limit.

    Returns raw CAGR, or -999.0 for degenerate/rejected curves.
    """
    metrics = compute_metrics(equity)
    if metrics["n_days"] < 5:  # Relaxed from 60
        return -999.0
    if metrics["max_drawdown"] > max_dd_limit:
        return -999.0
    cagr = metrics["cagr"]
    if cagr <= 0:
        return -999.0
    return cagr


# Max-profit param/metric keys for evaluate_combo
_MP_PARAM_KEYS = [
    "kama_period", "lookback_period", "top_n", "kama_buffer",
    "use_risk_adjusted", "sizing_mode",
]
_MP_METRIC_KEYS = [
    "total_return", "cagr", "max_drawdown", "sharpe", "calmar",
    "annualized_vol", "win_rate",
]
_MP_USER_ATTR_KEYS = _MP_METRIC_KEYS + [
    "use_risk_adjusted", "sizing_mode",
]


# ---------------------------------------------------------------------------
# Main search
# ---------------------------------------------------------------------------
def run_max_profit_search(
    close_prices: pd.DataFrame,
    open_prices: pd.DataFrame,
    tickers: list[str],
    initial_capital: float = INITIAL_CAPITAL,
    universe: str = "sp500",
    default_params: StrategyParams | None = None,
    space: dict[str, dict] | None = None,
    fixed_params: dict | None = None,
    n_trials: int = DEFAULT_MAX_PROFIT_TRIALS,
    n_workers: int | None = None,
    max_dd_limit: float = 0.60,
) -> MaxProfitResult:
    """Search parameter space for maximum CAGR using Optuna TPE.

    1. Pre-compute KAMA for all possible kama_period values (parallel).
    2. Use Optuna TPE sampler to explore the parameter space efficiently.
    3. Evaluate each combination in parallel via ProcessPoolExecutor.
    4. Return results sorted by CAGR.
    """
    default_params = default_params or StrategyParams()
    space = space or MAX_PROFIT_SPACE
    if not n_workers or n_workers < 1:
        n_workers = os.cpu_count()

    log.info(
        "max_profit_start",
        universe=universe,
        n_trials=n_trials,
        n_workers=n_workers,
    )

    # Pre-compute KAMA for all possible kama_period values
    kama_periods = _get_kama_periods_from_space(space)
    kama_caches = precompute_kama_caches(
        close_prices, tickers, kama_periods, n_workers,
    )
    log.info("kama_precomputed", n_periods=len(kama_caches))

    # Run default params to get baseline
    default_kama = kama_caches.get(default_params.kama_period)
    default_result = run_simulation(
        close_prices, open_prices, tickers, initial_capital,
        params=default_params, kama_cache=default_kama,
    )
    default_metrics = compute_metrics(default_result.equity)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    # Import fresh references to avoid stale function objects after Streamlit
    # module reloads (prevents PicklingError with ProcessPoolExecutor).
    from src.portfolio_sim.parallel import evaluate_combo, init_eval_worker

    # Use ask/tell API with batch parallelism via ProcessPoolExecutor.
    executor = ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=init_eval_worker,
        initargs=(close_prices, open_prices, tickers, initial_capital, kama_caches),
    )

    pbar = tqdm(total=n_trials, desc=f"Optuna search ({universe})", unit="trial")
    trials_done = 0

    while trials_done < n_trials:
        batch_size = min(n_workers, n_trials - trials_done)
        trials = [study.ask() for _ in range(batch_size)]
        params_list = [suggest_params(t, space, fixed_params) for t in trials]

        futures = {
            executor.submit(
                evaluate_combo,
                (p, max_dd_limit, compute_cagr_objective, "objective_cagr",
                 _MP_PARAM_KEYS, _MP_METRIC_KEYS),
            ): (t, p)
            for t, p in zip(trials, params_list)
        }
        for future in as_completed(futures):
            trial, _ = futures[future]
            result_dict = future.result()
            obj = result_dict["objective_cagr"]
            value = obj if obj > -999.0 else float("-inf")
            for key in _MP_USER_ATTR_KEYS:
                trial.set_user_attr(key, result_dict.get(key, 0.0))
            study.tell(trial, value)
            pbar.update(1)
            trials_done += 1

    pbar.close()
    executor.shutdown(wait=True)

    # Extract results into DataFrame
    rows: list[dict] = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            row = dict(trial.params)
            obj_val = trial.value
            row["objective_cagr"] = obj_val if obj_val != float("-inf") else -999.0
            row.update(trial.user_attrs)
            rows.append(row)

    grid_df = pd.DataFrame(rows)

    log.info(
        "max_profit_done",
        universe=universe,
        n_valid=int((grid_df["objective_cagr"] > -999.0).sum()),
        n_total=len(grid_df),
    )

    return MaxProfitResult(
        universe=universe,
        grid_results=grid_df,
        default_metrics=default_metrics,
        default_params=default_params,
    )


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------
def format_max_profit_report(
    result: MaxProfitResult,
    top_n: int = 20,
) -> str:
    """Format a human-readable max-profit search report."""
    lines: list[str] = []
    lines.append("=" * 90)
    lines.append(f"MAXIMUM PROFIT SEARCH — {result.universe.upper()} Universe")
    lines.append("=" * 90)

    # Default params baseline
    bp = result.default_params
    dm = result.default_metrics
    lines.append("")
    lines.append("Default Parameters Baseline:")
    lines.append(
        f"  kama_period={bp.kama_period}, lookback_period={bp.lookback_period}, "
        f"top_n={bp.top_n}, kama_buffer={bp.kama_buffer}"
    )
    lines.append(
        f"  risk_adjusted={bp.use_risk_adjusted}, sizing={bp.sizing_mode}"
    )
    lines.append(
        f"  CAGR: {dm['cagr']:.2%}  Total Return: {dm['total_return']:.2%}  "
        f"Max DD: {dm['max_drawdown']:.2%}  Sharpe: {dm['sharpe']:.2f}"
    )

    # Trials summary
    grid_df = result.grid_results
    valid = grid_df[grid_df["objective_cagr"] > -999.0]
    lines.append("")
    lines.append(
        f"Trials: {len(grid_df)} evaluated, "
        f"{len(valid)} valid ({len(valid) / len(grid_df):.0%})"
    )

    if not valid.empty:
        lines.append(
            f"  CAGR range: {valid['cagr'].min():.2%} .. {valid['cagr'].max():.2%}"
        )
        lines.append(f"  Median CAGR: {valid['cagr'].median():.2%}")

    # Top combos
    lines.append("")
    lines.append("-" * 90)
    lines.append(f"Top {top_n} Combinations by CAGR:")
    lines.append("-" * 90)

    if not valid.empty:
        top = valid.nlargest(top_n, "cagr")
        header = (
            f"  {'#':>3} {'kama':>5} {'lbk':>5} {'top_n':>5} {'buf':>7} "
            f"{'rsk_adj':>7} {'sizing':>12} "
            f"{'CAGR':>8} {'Return':>9} {'MaxDD':>8} {'Sharpe':>7}"
        )
        lines.append(header)
        lines.append("  " + "-" * 88)
        for rank, (_, row) in enumerate(top.iterrows(), 1):
            lines.append(
                f"  {rank:>3} {int(row['kama_period']):>5} "
                f"{int(row['lookback_period']):>5} "
                f"{int(row['top_n']):>5} {row['kama_buffer']:>7.4f} "
                f"{'Y' if row['use_risk_adjusted'] else 'N':>7} "
                f"{row['sizing_mode']:>12} "
                f"{row['cagr']:>8.2%} {row['total_return']:>9.2%} "
                f"{row['max_drawdown']:>8.2%} {row['sharpe']:>7.2f}"
            )

    # Best vs default comparison
    if not valid.empty:
        best = valid.nlargest(1, "cagr").iloc[0]
        lines.append("")
        lines.append("-" * 90)
        lines.append("Best vs Default:")
        lines.append("-" * 90)
        lines.append(
            f"  Default CAGR: {dm['cagr']:>8.2%}   |   Best CAGR: {best['cagr']:>8.2%}   "
            f"|   Improvement: {best['cagr'] - dm['cagr']:>+8.2%}"
        )
        lines.append(
            f"  Default Return: {dm['total_return']:>7.2%}  |   Best Return: {best['total_return']:>7.2%}"
        )
        # Compute dollar profit from $10,000
        default_profit = 10_000 * dm["total_return"]
        best_profit = 10_000 * best["total_return"]
        lines.append(
            f"  Default Profit: ${default_profit:>10,.0f}  "
            f"|   Best Profit: ${best_profit:>10,.0f}"
        )

    lines.append("")
    lines.append("=" * 90)
    return "\n".join(lines)


def select_best_from_search(result: MaxProfitResult) -> StrategyParams | None:
    """Select the trial with the best CAGR from a single-objective search."""
    grid = result.grid_results
    if grid.empty:
        return None

    valid = grid[grid["objective_cagr"] > -999.0]
    if valid.empty:
        return None

    best = valid.loc[valid["objective_cagr"].idxmax()]

    kwargs = {}
    for key in _MP_PARAM_KEYS:
        if key in best.index:
            kwargs[key] = best[key]
    return StrategyParams(**kwargs)
