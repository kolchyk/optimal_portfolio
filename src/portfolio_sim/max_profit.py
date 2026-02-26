"""Maximum profit parameter search for KAMA momentum strategy.

Unlike the sensitivity analysis in optimizer.py (which checks robustness),
this module searches for the parameter combination that maximizes CAGR
over a given period using Optuna's TPE sampler. The drawdown rejection
limit is relaxed to 60%.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import optuna
import pandas as pd
import structlog

from src.portfolio_sim.config import (
    DEFAULT_MAX_PROFIT_TRIALS,
    INITIAL_CAPITAL,
    MAX_PROFIT_SPACE,
)
from src.portfolio_sim.engine import run_simulation
from src.portfolio_sim.optimizer import (
    _get_kama_periods_from_space,
    make_objective,
    precompute_kama_caches,
)
from src.portfolio_sim.parallel import run_optuna_batch_loop, select_best_params
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
    """CAGR-maximizing objective with relaxed drawdown limit."""
    return make_objective("cagr", max_dd_limit=max_dd_limit, min_n_days=5)(equity)


# Metric keys tracked per trial
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
    eval_start: str | None = None,
    eval_end: str | None = None,
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
    
    default_equity = default_result.equity
    if eval_start:
        default_equity = default_equity.loc[eval_start:]
    if eval_end:
        default_equity = default_equity.loc[:eval_end]
    default_metrics = compute_metrics(default_equity)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    # Import fresh references to avoid stale function objects after Streamlit
    # module reloads (prevents PicklingError with ProcessPoolExecutor).
    from src.portfolio_sim.parallel import init_eval_worker

    # Use ask/tell API with batch parallelism via ProcessPoolExecutor.
    executor = ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=init_eval_worker,
        initargs=(close_prices, open_prices, tickers, initial_capital, kama_caches),
    )

    slice_spec = {
        "eval_start": eval_start,
        "eval_end": eval_end,
        "tickers": tickers,
    }

    grid_df = run_optuna_batch_loop(
        study, executor,
        space=space,
        n_trials=n_trials,
        n_workers=n_workers,
        objective_fn=compute_cagr_objective,
        max_dd_limit=max_dd_limit,
        objective_key="objective_cagr",
        param_keys=ALL_PARAM_NAMES,
        metric_keys=_MP_METRIC_KEYS,
        user_attr_keys=_MP_USER_ATTR_KEYS,
        fixed_params=fixed_params,
        slice_spec=slice_spec,
        desc=f"Optuna search ({universe})",
    )

    executor.shutdown(wait=True)

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
    if result.grid_results.empty:
        return None
    return select_best_params(result.grid_results, "objective_cagr", ALL_PARAM_NAMES)
