"""Parameter optimization utilities.

Shared infrastructure for KAMA pre-computation and strategy optimization.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial

import optuna
import pandas as pd
import structlog
from tqdm import tqdm

from src.portfolio_sim.config import (
    INITIAL_CAPITAL,
    MAX_DD_LIMIT,
    PARAM_NAMES,
    SEARCH_SPACE,
    SPY_TICKER,
    get_kama_periods,
)
from src.portfolio_sim.indicators import compute_kama_series
from src.portfolio_sim.params import StrategyParams
from src.portfolio_sim.reporting import compute_metrics

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Shared data for KAMA worker processes
# ---------------------------------------------------------------------------
_kama_shared: dict = {}


def _clamp_to_space(value, spec: dict):
    """Clamp a parameter value to the nearest valid value in a search space spec."""
    if spec["type"] == "categorical":
        if value in spec["choices"]:
            return value
        return spec["choices"][0]
    low, high = spec["low"], spec["high"]
    step = spec.get("step")
    clamped = max(low, min(high, value))
    if step:
        clamped = low + round((clamped - low) / step) * step
        clamped = max(low, min(high, clamped))
    return type(value)(clamped) if spec["type"] == "int" else clamped


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class SensitivityResult:
    """Results from parameter optimization."""

    grid_results: pd.DataFrame
    base_params: StrategyParams
    base_objective: float


# ---------------------------------------------------------------------------
# Objective functions
# ---------------------------------------------------------------------------
def make_objective(
    metric: str = "total_return",
    max_dd_limit: float = 0.30,
    min_n_days: int = 60,
):
    """Factory that returns an objective function for Optuna trials."""
    def _objective(equity: pd.Series) -> float:
        metrics = compute_metrics(equity)
        if metrics["n_days"] < min_n_days:
            return -999.0
        if metrics["max_drawdown"] > max_dd_limit:
            return -999.0
        value = metrics[metric]
        if value <= 0:
            return -999.0
        return value
    return _objective


def _raw_metric_objective(
    equity: pd.Series,
    max_dd_limit: float,
    metric: str = "total_return",
) -> float:
    """Raw metric objective — returns metric value."""
    return compute_metrics(equity)[metric]


# ---------------------------------------------------------------------------
# KAMA pre-computation (parallel)
# ---------------------------------------------------------------------------
def _init_kama_worker(close_prices: pd.DataFrame):
    """Initializer for KAMA worker processes."""
    _kama_shared["close"] = close_prices


def _compute_single_kama(args: tuple[int, str]) -> tuple[int, str, pd.Series]:
    """Compute KAMA for a single (period, ticker) pair using shared data."""
    period, ticker = args
    series = _kama_shared["close"][ticker].dropna()
    return period, ticker, compute_kama_series(series, period=period)


def precompute_kama_caches(
    close_prices: pd.DataFrame,
    tickers: list[str],
    kama_periods: list[int],
    n_workers: int | None = None,
) -> dict[int, dict[str, pd.Series]]:
    """Precompute KAMA series for all unique kama_period values (parallel).

    Returns:
        {kama_period: {ticker: kama_series}}
    """
    all_tickers = [t for t in set(tickers + [SPY_TICKER]) if t in close_prices.columns]
    unique_periods = sorted(set(kama_periods))
    if not n_workers or n_workers < 1:
        n_workers = os.cpu_count()

    tasks = [(p, t) for p in unique_periods for t in all_tickers]

    result: dict[int, dict[str, pd.Series]] = {p: {} for p in unique_periods}

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_kama_worker,
        initargs=(close_prices,),
    ) as executor:
        futures = {executor.submit(_compute_single_kama, task): task for task in tasks}
        for future in tqdm(as_completed(futures), total=len(tasks), desc="KAMA precompute", unit="series"):
            period, ticker, kama_series = future.result()
            result[period][ticker] = kama_series

    return result


# ---------------------------------------------------------------------------
# Main optimization
# ---------------------------------------------------------------------------
_METRIC_KEYS = ["cagr", "max_drawdown", "sharpe", "calmar"]


def run_sensitivity(
    close_prices: pd.DataFrame,
    open_prices: pd.DataFrame,
    tickers: list[str],
    initial_capital: float = INITIAL_CAPITAL,
    base_params: StrategyParams | None = None,
    space: dict[str, dict] | None = None,
    n_trials: int = 100,
    n_workers: int | None = None,
    max_dd_limit: float = MAX_DD_LIMIT,
    min_n_days: int = 60,
    metric: str = "total_return",
    kama_caches: dict[int, dict[str, pd.Series]] | None = None,
    executor: ProcessPoolExecutor | None = None,
    verbose: bool = True,
) -> SensitivityResult:
    """Run parameter optimisation using Optuna TPE sampler."""
    base_params = base_params or StrategyParams()
    space = space or SEARCH_SPACE
    if not n_workers or n_workers < 1:
        n_workers = os.cpu_count()

    log.info(
        "sensitivity_start",
        n_trials=n_trials,
        n_workers=n_workers,
    )

    if kama_caches is None:
        kama_periods = get_kama_periods(space)
        kama_caches = precompute_kama_caches(
            close_prices, tickers, kama_periods, n_workers,
        )
        log.info("kama_precomputed", n_periods=len(kama_caches))

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    # Enqueue base params as first trial
    study.enqueue_trial({
        name: _clamp_to_space(getattr(base_params, name), spec)
        for name, spec in space.items()
    })

    from src.portfolio_sim.parallel import init_eval_worker, run_optuna_batch_loop

    own_executor = executor is None

    if own_executor:
        executor = ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=init_eval_worker,
            initargs=(close_prices, open_prices, tickers, initial_capital, kama_caches),
        )
        slice_spec = None
    else:
        slice_spec = {
            "is_start": close_prices.index[0],
            "is_end": close_prices.index[-1],
            "tickers": list(tickers),
        }

    objective_fn = partial(_raw_metric_objective, metric=metric)

    grid_df = run_optuna_batch_loop(
        study, executor,
        space=space,
        n_trials=n_trials,
        n_workers=n_workers,
        objective_fn=objective_fn,
        max_dd_limit=max_dd_limit,
        objective_key="objective",
        param_keys=PARAM_NAMES,
        metric_keys=_METRIC_KEYS,
        slice_spec=slice_spec,
        desc="Sensitivity trials",
        verbose=verbose,
    )

    if own_executor:
        executor.shutdown(wait=True)

    # Compute base params objective
    mask = pd.Series(True, index=grid_df.index)
    for name in PARAM_NAMES:
        if name in grid_df.columns:
            mask = mask & (grid_df[name] == getattr(base_params, name))
    base_row = grid_df[mask]
    if not base_row.empty:
        base_objective = float(base_row.iloc[0]["objective"])
    else:
        base_objective = float("nan")

    log.info(
        "sensitivity_done",
        n_valid=int((grid_df["objective"] > -999.0).sum()),
        n_total=len(grid_df),
    )

    return SensitivityResult(
        grid_results=grid_df,
        base_params=base_params,
        base_objective=base_objective,
    )


def find_best_params(result: SensitivityResult) -> StrategyParams | None:
    """Extract the best parameter combo from optimisation results."""
    from src.portfolio_sim.parallel import select_best_params
    return select_best_params(result.grid_results, "objective", PARAM_NAMES)
