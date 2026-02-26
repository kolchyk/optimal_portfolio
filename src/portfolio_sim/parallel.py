"""Shared parallel evaluation infrastructure for Optuna-based searches.

Used by both optimizer.py (sensitivity analysis) and max_profit.py
(CAGR-maximizing search) to avoid duplicating worker initializers,
evaluation functions, and parameter suggestion logic.
"""

from __future__ import annotations

from typing import Callable

import optuna
import pandas as pd

from src.portfolio_sim.engine import run_simulation
from src.portfolio_sim.params import StrategyParams
from src.portfolio_sim.reporting import compute_metrics

# ---------------------------------------------------------------------------
# Shared data for worker processes (initializer pattern avoids repeated
# pickling of large DataFrames — sent once per worker, not once per task).
# ---------------------------------------------------------------------------
_shared: dict = {}


def init_eval_worker(
    close_prices: pd.DataFrame,
    open_prices: pd.DataFrame,
    tickers: list[str],
    initial_capital: float,
    kama_caches: dict[int, dict[str, pd.Series]],
):
    """Initializer for evaluation worker processes."""
    _shared["close"] = close_prices
    _shared["open"] = open_prices
    _shared["tickers"] = tickers
    _shared["capital"] = initial_capital
    _shared["kama_caches"] = kama_caches


def evaluate_combo(
    args: tuple,
) -> dict:
    """Evaluate a single param combo, optionally on a date-sliced subset.

    Args:
        args: Tuple of (params, max_dd_limit, objective_fn, objective_key,
              param_keys, metric_keys) or
              (params, max_dd_limit, objective_fn, objective_key,
              param_keys, metric_keys, slice_spec).

            - params: strategy parameters to evaluate
            - max_dd_limit: passed to the objective function
            - objective_fn: callable(equity, max_dd_limit) -> float
            - objective_key: key name for the objective value in result dict
            - param_keys: param attribute names to include in result
            - metric_keys: metric dict keys to include in result
            - slice_spec (optional): dict with "is_start", "is_end",
              "tickers" — workers slice shared data to the given range

    Returns:
        Dict with parameter values, objective, and requested metrics.
    """
    if len(args) == 7:
        params, max_dd_limit, objective_fn, objective_key, param_keys, metric_keys, slice_spec = args
    else:
        params, max_dd_limit, objective_fn, objective_key, param_keys, metric_keys = args
        slice_spec = None

    try:
        close = _shared["close"]
        open_ = _shared["open"]
        tickers = _shared["tickers"]

        if slice_spec is not None:
            close = close.loc[slice_spec["is_start"]:slice_spec["is_end"]]
            open_ = open_.loc[slice_spec["is_start"]:slice_spec["is_end"]]
            tickers = slice_spec["tickers"]

        kama_cache = _shared["kama_caches"].get(params.kama_period)
        result = run_simulation(
            close, open_, tickers,
            _shared["capital"], params=params, kama_cache=kama_cache,
        )
        equity = result.equity
        if equity.empty:
            obj = -999.0
            metrics = {}
        else:
            obj = objective_fn(equity, max_dd_limit)
            metrics = compute_metrics(equity)
    except (ValueError, KeyError):
        obj = -999.0
        metrics = {}

    row: dict = {}
    for key in param_keys:
        row[key] = getattr(params, key)
    row[objective_key] = obj
    for key in metric_keys:
        row[key] = metrics.get(key, 0.0)
    return row


def suggest_params(
    trial: optuna.Trial,
    space: dict[str, dict],
    fixed_params: dict | None = None,
) -> StrategyParams:
    """Suggest a StrategyParams from an Optuna trial, with optional fixed overrides."""
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
