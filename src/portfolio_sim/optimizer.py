"""Parameter optimization for KAMA momentum strategy.

Uses Optuna TPE sampler to explore the parameter space within each
walk-forward optimization (WFO) in-sample window.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial

import numpy as np
import optuna
import pandas as pd
import structlog
from tqdm import tqdm

from src.portfolio_sim.config import (
    DEFAULT_N_TRIALS,
    INITIAL_CAPITAL,
    SEARCH_SPACE,
    SPY_TICKER,
)
from src.portfolio_sim.indicators import compute_kama_series
from src.portfolio_sim.parallel import (
    _shared,
    run_optuna_batch_loop,
    select_best_params,
)
from src.portfolio_sim.params import StrategyParams
from src.portfolio_sim.reporting import compute_metrics

log = structlog.get_logger()


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
PARAM_NAMES: list[str] = [
    "kama_period", "lookback_period", "kama_buffer", "top_n",
    "oos_days", "corr_threshold",
]


@dataclass
class SensitivityResult:
    """Results from parameter optimization."""

    grid_results: pd.DataFrame
    """All trial results: columns = param names + objective + metrics."""

    base_params: StrategyParams
    base_objective: float


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------
def make_objective(
    metric: str = "total_return",
    max_dd_limit: float = 0.30,
    min_n_days: int = 60,
):
    """Factory that returns an objective function for Optuna trials.

    The returned callable accepts an equity ``pd.Series`` and returns
    the chosen *metric* value, or ``-999.0`` when the equity curve is
    rejected (too short, excessive drawdown, or non-positive metric).
    """
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


def compute_objective(
    equity: pd.Series,
    max_dd_limit: float = 0.30,
    min_n_days: int = 60,
) -> float:
    """Sharpe ratio with drawdown cap and hard rejection.

    Returns:
        Sharpe ratio, or -999.0 when MaxDD exceeds *max_dd_limit*
        or the equity curve is degenerate.
    """
    return make_objective("sharpe", max_dd_limit, min_n_days)(equity)


# ---------------------------------------------------------------------------
# KAMA pre-computation (parallel)
# ---------------------------------------------------------------------------
def _init_kama_worker(close_prices: pd.DataFrame):
    """Initializer for KAMA worker processes."""
    _shared["close"] = close_prices


def _compute_single_kama(args: tuple[int, str]) -> tuple[int, str, pd.Series]:
    """Compute KAMA for a single (period, ticker) pair using shared data."""
    period, ticker = args
    series = _shared["close"][ticker].dropna()
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

    # Build all (period, ticker) tasks
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


def _get_kama_periods_from_space(space: dict[str, dict]) -> list[int]:
    """Extract all possible kama_period values from a search space definition."""
    spec = space.get("kama_period", {})
    if spec.get("type") == "categorical":
        return list(spec["choices"])
    return list(range(
        spec.get("low", 10),
        spec.get("high", 40) + 1,
        spec.get("step", 5),
    ))


# Metric keys tracked per trial
_SENS_METRIC_KEYS = ["cagr", "max_drawdown", "sharpe", "calmar"]


# ---------------------------------------------------------------------------
# Main optimization
# ---------------------------------------------------------------------------
def run_sensitivity(
    close_prices: pd.DataFrame,
    open_prices: pd.DataFrame,
    tickers: list[str],
    initial_capital: float = INITIAL_CAPITAL,
    base_params: StrategyParams | None = None,
    space: dict[str, dict] | None = None,
    n_trials: int = DEFAULT_N_TRIALS,
    n_workers: int | None = None,
    max_dd_limit: float = 0.30,
    min_n_days: int = 60,
    kama_caches: dict[int, dict[str, pd.Series]] | None = None,
    executor: ProcessPoolExecutor | None = None,
) -> SensitivityResult:
    """Run parameter optimization using Optuna TPE sampler.

    1. Pre-compute KAMA for all possible kama_period values (parallel).
    2. Use Optuna to sample parameter combinations efficiently.
    3. Evaluate each combination in parallel via ProcessPoolExecutor.

    When *kama_caches* is provided, KAMA pre-computation is skipped.
    When *executor* is provided, a shared ProcessPoolExecutor is reused
    instead of creating a new one (workers must hold full-range data;
    per-step slicing is handled via slice_spec in evaluate_combo).
    """
    base_params = base_params or StrategyParams()
    space = space or SEARCH_SPACE
    if not n_workers or n_workers < 1:
        n_workers = os.cpu_count()

    log.info(
        "sensitivity_start",
        n_trials=n_trials,
        n_workers=n_workers,
    )

    # Pre-compute KAMA for all possible kama_period values (skip if provided)
    if kama_caches is None:
        kama_periods = _get_kama_periods_from_space(space)
        kama_caches = precompute_kama_caches(close_prices, tickers, kama_periods, n_workers)
        log.info("kama_precomputed", n_periods=len(kama_caches))

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    # Enqueue base params as first trial to guarantee base_objective.
    # Clamp values to the search space so enqueue never fails when
    # base_params fall outside a narrow custom space.
    study.enqueue_trial({
        name: _clamp_to_space(getattr(base_params, name), spec)
        for name, spec in space.items()
    })

    # Import fresh references to avoid stale function objects after Streamlit
    # module reloads (prevents PicklingError with ProcessPoolExecutor).
    from src.portfolio_sim.parallel import init_eval_worker

    # Determine whether we own the executor (and must shut it down).
    own_executor = executor is None

    if own_executor:
        executor = ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=init_eval_worker,
            initargs=(close_prices, open_prices, tickers, initial_capital, kama_caches),
        )
        slice_spec = None  # workers already hold the correct data
    else:
        # External executor holds full-range data; build slice_spec so
        # workers slice to the date range and tickers for this call.
        slice_spec = {
            "is_start": close_prices.index[0],
            "is_end": close_prices.index[-1],
            "tickers": list(tickers),
        }

    objective_fn = partial(compute_objective, min_n_days=min_n_days)

    grid_df = run_optuna_batch_loop(
        study, executor,
        space=space,
        n_trials=n_trials,
        n_workers=n_workers,
        objective_fn=objective_fn,
        max_dd_limit=max_dd_limit,
        objective_key="objective",
        param_keys=PARAM_NAMES,
        metric_keys=_SENS_METRIC_KEYS,
        slice_spec=slice_spec,
        desc="Sensitivity trials",
    )

    if own_executor:
        executor.shutdown(wait=True)

    # Compute base params objective — match on all params present in grid
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
    """Extract the best parameter combo from optimization results.

    Returns StrategyParams for the combo with highest objective,
    or None if no valid combo was found.
    """
    return select_best_params(result.grid_results, "objective", PARAM_NAMES)
