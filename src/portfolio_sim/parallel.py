"""Shared parallel evaluation infrastructure for Optuna-based searches.

Used by both optimizer.py (sensitivity analysis) and max_profit.py
(CAGR-maximizing search) to avoid duplicating worker initializers,
evaluation functions, and parameter suggestion logic.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable

import optuna
import pandas as pd
from tqdm import tqdm

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
            # Slicing for simulation (must include warmup)
            sim_start = slice_spec.get("sim_start")
            sim_end = slice_spec.get("sim_end")
            if sim_start:
                close = close.loc[sim_start:]
                open_ = open_.loc[sim_start:]
            if sim_end:
                close = close.loc[:sim_end]
                open_ = open_.loc[:sim_end]
            
            # Use original tickers if not overridden
            tickers = slice_spec.get("tickers", tickers)

        kama_cache = _shared["kama_caches"].get(params.kama_period)
        spy_kama_cache = _shared["kama_caches"].get(params.kama_spy_period)
        spy_kama_series = spy_kama_cache.get("SPY") if spy_kama_cache else None
        result = run_simulation(
            close, open_, tickers,
            _shared["capital"], params=params, kama_cache=kama_cache,
            spy_kama_series=spy_kama_series,
        )
        equity = result.equity

        # Slicing for evaluation (objective + metrics)
        if slice_spec is not None:
            eval_start = slice_spec.get("eval_start")
            eval_end = slice_spec.get("eval_end")
            if eval_start and not equity.empty:
                # Use nearest available date if exact match not found
                equity = equity.loc[eval_start:]
            if eval_end and not equity.empty:
                equity = equity.loc[:eval_end]

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


def run_optuna_batch_loop(
    study: optuna.Study,
    executor: ProcessPoolExecutor,
    *,
    space: dict[str, dict],
    n_trials: int,
    n_workers: int,
    objective_fn: Callable,
    max_dd_limit: float,
    objective_key: str,
    param_keys: list[str],
    metric_keys: list[str],
    user_attr_keys: list[str] | None = None,
    fixed_params: dict | None = None,
    slice_spec: dict | None = None,
    desc: str = "Optuna trials",
) -> pd.DataFrame:
    """Run batch-parallel Optuna ask/tell loop and return results DataFrame.

    Parameters
    ----------
    study : optuna.Study
        Pre-created Optuna study (direction, sampler, enqueued trials).
    executor : ProcessPoolExecutor
        Already-initialized pool with ``init_eval_worker``.
    space : dict
        Parameter search space for ``suggest_params``.
    n_trials, n_workers : int
        Total trials and batch parallelism.
    objective_fn, max_dd_limit, objective_key : ...
        Passed through to ``evaluate_combo``.
    param_keys, metric_keys : list[str]
        Which parameter/metric keys to record per trial.
    user_attr_keys : list[str] | None
        Keys to store as Optuna user attrs.  Defaults to *metric_keys*.
    fixed_params : dict | None
        Fixed overrides for ``suggest_params``.
    slice_spec : dict | None
        Optional date/ticker slicing spec for workers.
    desc : str
        Progress bar description.

    Returns
    -------
    pd.DataFrame
        One row per completed trial with param values, objective, and metrics.
    """
    if user_attr_keys is None:
        user_attr_keys = metric_keys

    pbar = tqdm(total=n_trials, desc=desc, unit="trial")
    trials_done = 0

    while trials_done < n_trials:
        batch_size = min(n_workers, n_trials - trials_done)
        trials = [study.ask() for _ in range(batch_size)]
        params_list = [suggest_params(t, space, fixed_params) for t in trials]

        combo = lambda p: (
            p, max_dd_limit, objective_fn, objective_key,
            param_keys, metric_keys,
        )

        futures = {
            executor.submit(
                evaluate_combo,
                combo(p) if slice_spec is None else combo(p) + (slice_spec,),
            ): (t, p)
            for t, p in zip(trials, params_list)
        }
        for future in as_completed(futures):
            trial, _ = futures[future]
            result_dict = future.result()
            obj = result_dict[objective_key]
            value = obj if obj > -999.0 else float("-inf")
            for key in user_attr_keys:
                trial.set_user_attr(key, result_dict.get(key, 0.0))
            study.tell(trial, value)
            pbar.update(1)
            trials_done += 1

    pbar.close()

    # Extract results into DataFrame
    rows: list[dict] = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            row = dict(trial.params)
            obj_val = trial.value
            row[objective_key] = obj_val if obj_val != float("-inf") else -999.0
            row.update(trial.user_attrs)
            rows.append(row)

    return pd.DataFrame(rows)


def select_best_params(
    grid_df: pd.DataFrame,
    objective_col: str,
    param_keys: list[str],
) -> StrategyParams | None:
    """Select the best parameter combo from an optimization grid.

    Returns the ``StrategyParams`` for the row with the highest
    *objective_col* value (excluding rejected trials with -999.0),
    or ``None`` if no valid combo was found.

    Values are cast to native Python types (int/float/str/bool) so that
    downstream code (e.g. ``pd.rolling()``) receives proper types instead
    of numpy scalars.
    """
    valid = grid_df[grid_df[objective_col] > -999.0]
    if valid.empty:
        return None
    best = valid.loc[valid[objective_col].idxmax()]

    # Map StrategyParams field names → expected types
    _field_types = {f.name: f.type for f in StrategyParams.__dataclass_fields__.values()}

    kwargs = {}
    for key in param_keys:
        if key in best.index:
            val = best[key]
            expected = _field_types.get(key)
            if expected == "int" or expected is int:
                val = int(val)
            elif expected == "float" or expected is float:
                val = float(val)
            kwargs[key] = val
    return StrategyParams(**kwargs)
