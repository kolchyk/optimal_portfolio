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
from tqdm import tqdm

from src.portfolio_sim.config import INITIAL_CAPITAL
from src.portfolio_sim.engine import run_simulation
from src.portfolio_sim.optimizer import (
    _get_kama_periods_from_space,
    precompute_kama_caches,
)
from src.portfolio_sim.params import StrategyParams
from src.portfolio_sim.reporting import compute_metrics

log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Shared data for worker processes (initializer pattern)
# ---------------------------------------------------------------------------
_shared: dict = {}

# ---------------------------------------------------------------------------
# Default search space for max profit search
# ---------------------------------------------------------------------------
MAX_PROFIT_SPACE: dict[str, dict] = {
    "kama_period": {"type": "categorical", "choices": [5, 10, 15, 20, 30]},
    "lookback_period": {"type": "int", "low": 30, "high": 200, "step": 10},
    "top_n": {"type": "int", "low": 3, "high": 15, "step": 2},
    "kama_buffer": {"type": "float", "low": 0.003, "high": 0.02, "step": 0.001},
    "use_risk_adjusted": {"type": "categorical", "choices": [True, False]},
    "enable_regime_filter": {"type": "categorical", "choices": [True, False]},
    "sizing_mode": {"type": "categorical", "choices": ["equal_weight", "risk_parity"]},
}

DEFAULT_MAX_PROFIT_TRIALS: int = 500

ALL_PARAM_NAMES: list[str] = [
    "kama_period", "lookback_period", "top_n", "kama_buffer",
    "use_risk_adjusted", "enable_regime_filter", "sizing_mode",
    "enable_correlation_filter",
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
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
    if metrics["n_days"] < 60:
        return -999.0
    if metrics["max_drawdown"] > max_dd_limit:
        return -999.0
    cagr = metrics["cagr"]
    if cagr <= 0:
        return -999.0
    return cagr


# ---------------------------------------------------------------------------
# Search space helper with fixed param overrides
# ---------------------------------------------------------------------------
def _suggest_max_profit_params(
    trial: optuna.Trial,
    space: dict[str, dict],
    fixed_params: dict | None = None,
) -> StrategyParams:
    """Suggest params from trial, with fixed overrides.

    Parameters in fixed_params are not suggested by Optuna but set directly.
    """
    fixed_params = fixed_params or {}
    kwargs = {}
    for name, spec in space.items():
        if name in fixed_params:
            kwargs[name] = fixed_params[name]
            continue
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
    kwargs.update(fixed_params)
    return StrategyParams(**kwargs)


# ---------------------------------------------------------------------------
# Parallel evaluation helpers
# ---------------------------------------------------------------------------
def _init_max_profit_worker(
    close_prices: pd.DataFrame,
    open_prices: pd.DataFrame,
    tickers: list[str],
    initial_capital: float,
    kama_caches: dict[int, dict[str, pd.Series]],
):
    """Initializer for max-profit search workers."""
    _shared["close"] = close_prices
    _shared["open"] = open_prices
    _shared["tickers"] = tickers
    _shared["capital"] = initial_capital
    _shared["kama_caches"] = kama_caches


def _evaluate_max_profit_combo(args: tuple[StrategyParams, float]) -> dict:
    """Evaluate a single param combo, returning CAGR as the objective."""
    params, max_dd_limit = args
    try:
        kama_cache = _shared["kama_caches"].get(params.kama_period)
        result = run_simulation(
            _shared["close"], _shared["open"], _shared["tickers"],
            _shared["capital"], params=params, kama_cache=kama_cache,
        )
        equity = result.equity
        if equity.empty:
            obj = -999.0
            metrics = {}
        else:
            obj = compute_cagr_objective(equity, max_dd_limit=max_dd_limit)
            metrics = compute_metrics(equity)
    except (ValueError, KeyError):
        obj = -999.0
        metrics = {}

    return {
        "kama_period": params.kama_period,
        "lookback_period": params.lookback_period,
        "top_n": params.top_n,
        "kama_buffer": params.kama_buffer,
        "use_risk_adjusted": params.use_risk_adjusted,
        "enable_regime_filter": params.enable_regime_filter,
        "sizing_mode": params.sizing_mode,
        "enable_correlation_filter": params.enable_correlation_filter,
        "objective_cagr": obj,
        "total_return": metrics.get("total_return", 0.0),
        "cagr": metrics.get("cagr", 0.0),
        "max_drawdown": metrics.get("max_drawdown", 0.0),
        "sharpe": metrics.get("sharpe", 0.0),
        "calmar": metrics.get("calmar", 0.0),
        "annualized_vol": metrics.get("annualized_vol", 0.0),
        "win_rate": metrics.get("win_rate", 0.0),
    }


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
    n_workers = n_workers or max(1, os.cpu_count() - 1)

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

    # Create ProcessPoolExecutor with initializer (shared data pattern)
    executor = ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_max_profit_worker,
        initargs=(close_prices, open_prices, tickers, initial_capital, kama_caches),
    )

    pbar = tqdm(total=n_trials, desc=f"Optuna search ({universe})", unit="trial")

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_max_profit_params(trial, space, fixed_params)
        future = executor.submit(_evaluate_max_profit_combo, (params, max_dd_limit))
        result = future.result()
        for key in ("total_return", "cagr", "max_drawdown", "sharpe", "calmar",
                     "annualized_vol", "win_rate"):
            trial.set_user_attr(key, result.get(key, 0.0))
        for key in ("use_risk_adjusted", "enable_regime_filter", "sizing_mode",
                     "enable_correlation_filter"):
            trial.set_user_attr(key, result.get(key))
        pbar.update(1)
        obj = result["objective_cagr"]
        return obj if obj > -999.0 else float("-inf")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=n_workers)

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
        f"  risk_adjusted={bp.use_risk_adjusted}, regime_filter={bp.enable_regime_filter}, "
        f"sizing={bp.sizing_mode}"
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
            f"{'rsk_adj':>7} {'regime':>7} {'sizing':>12} "
            f"{'CAGR':>8} {'Return':>9} {'MaxDD':>8} {'Sharpe':>7}"
        )
        lines.append(header)
        lines.append("  " + "-" * 86)
        for rank, (_, row) in enumerate(top.iterrows(), 1):
            lines.append(
                f"  {rank:>3} {int(row['kama_period']):>5} "
                f"{int(row['lookback_period']):>5} "
                f"{int(row['top_n']):>5} {row['kama_buffer']:>7.4f} "
                f"{'Y' if row['use_risk_adjusted'] else 'N':>7} "
                f"{'Y' if row['enable_regime_filter'] else 'N':>7} "
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
