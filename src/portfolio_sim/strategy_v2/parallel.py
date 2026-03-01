"""V2 parallel evaluation infrastructure.

Mirrors parallel.py but uses run_simulation_v2 and StrategyParamsV2.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable

import optuna
import pandas as pd
from tqdm import tqdm

from src.portfolio_sim.reporting import compute_metrics
from src.portfolio_sim.strategy_v2.engine import run_simulation_v2
from src.portfolio_sim.strategy_v2.params import StrategyParamsV2

# ---------------------------------------------------------------------------
# Shared data for V2 worker processes
# ---------------------------------------------------------------------------
_shared_v2: dict = {}


def init_eval_worker_v2(
    close_prices: pd.DataFrame,
    open_prices: pd.DataFrame,
    tickers: list[str],
    initial_capital: float,
    kama_caches: dict[int, dict[str, pd.Series]],
):
    """Initialiser for V2 evaluation worker processes."""
    _shared_v2["close"] = close_prices
    _shared_v2["open"] = open_prices
    _shared_v2["tickers"] = tickers
    _shared_v2["capital"] = initial_capital
    _shared_v2["kama_caches"] = kama_caches


def evaluate_combo_v2(args: tuple) -> dict:
    """Evaluate a single V2 param combo, optionally on a date-sliced subset."""
    if len(args) == 7:
        params, max_dd_limit, objective_fn, objective_key, param_keys, metric_keys, slice_spec = args
    else:
        params, max_dd_limit, objective_fn, objective_key, param_keys, metric_keys = args
        slice_spec = None

    try:
        close = _shared_v2["close"]
        open_ = _shared_v2["open"]
        tickers = _shared_v2["tickers"]

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

        kama_cache = _shared_v2["kama_caches"].get(params.kama_period)
        result = run_simulation_v2(
            close, open_, tickers,
            _shared_v2["capital"], params=params, kama_cache=kama_cache,
        )
        equity = result.equity

        if slice_spec is not None:
            eval_start = slice_spec.get("eval_start")
            eval_end = slice_spec.get("eval_end")
            if eval_start and not equity.empty:
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


def suggest_params_v2(
    trial: optuna.Trial,
    space: dict[str, dict],
    fixed_params: dict | None = None,
) -> StrategyParamsV2:
    """Suggest a StrategyParamsV2 from an Optuna trial."""
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
    return StrategyParamsV2(**kwargs)


def run_optuna_batch_loop_v2(
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
    desc: str = "V2 Optuna trials",
    verbose: bool = True,
) -> pd.DataFrame:
    """Run batch-parallel Optuna ask/tell loop for V2 strategy."""
    if user_attr_keys is None:
        user_attr_keys = metric_keys

    pbar = tqdm(total=n_trials, desc=desc, unit="trial", disable=not verbose)
    trials_done = 0

    while trials_done < n_trials:
        batch_size = min(n_workers, n_trials - trials_done)
        trials = [study.ask() for _ in range(batch_size)]
        params_list = [suggest_params_v2(t, space, fixed_params) for t in trials]

        combo = lambda p: (
            p, max_dd_limit, objective_fn, objective_key,
            param_keys, metric_keys,
        )

        futures = {
            executor.submit(
                evaluate_combo_v2,
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

    rows: list[dict] = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            row = dict(trial.params)
            obj_val = trial.value
            row[objective_key] = obj_val if obj_val != float("-inf") else -999.0
            row.update(trial.user_attrs)
            rows.append(row)

    return pd.DataFrame(rows)


def select_best_params_v2(
    grid_df: pd.DataFrame,
    objective_col: str,
    param_keys: list[str],
) -> StrategyParamsV2 | None:
    """Select the best V2 parameter combo from an optimisation grid."""
    valid = grid_df[grid_df[objective_col] > -999.0]
    if valid.empty:
        return None
    best = valid.loc[valid[objective_col].idxmax()]

    _field_types = {
        f.name: f.type
        for f in StrategyParamsV2.__dataclass_fields__.values()
    }

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
    return StrategyParamsV2(**kwargs)
