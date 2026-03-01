"""S2 parallel evaluation infrastructure.

Mirrors strategy_v2/parallel.py but uses run_simulation_s2 and S2StrategyParams.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable

import optuna
import pandas as pd
from tqdm import tqdm

from src.portfolio_sim.reporting import compute_metrics
from src.portfolio_sim.strategy_s2.engine import run_simulation_s2
from src.portfolio_sim.strategy_s2.params import S2StrategyParams

# ---------------------------------------------------------------------------
# Shared data for S2 worker processes
# ---------------------------------------------------------------------------
_shared_s2: dict = {}


def init_eval_worker_s2(
    close_prices: pd.DataFrame,
    open_prices: pd.DataFrame,
    tickers: list[str],
    initial_capital: float,
    kama_caches: dict[int, dict[str, pd.Series]],
):
    """Initialiser for S2 evaluation worker processes."""
    _shared_s2["close"] = close_prices
    _shared_s2["open"] = open_prices
    _shared_s2["tickers"] = tickers
    _shared_s2["capital"] = initial_capital
    _shared_s2["kama_caches"] = kama_caches


def evaluate_combo_s2(args: tuple) -> dict:
    """Evaluate a single S2 param combo, optionally on a date-sliced subset."""
    if len(args) == 7:
        params, max_dd_limit, objective_fn, objective_key, param_keys, metric_keys, slice_spec = args
    else:
        params, max_dd_limit, objective_fn, objective_key, param_keys, metric_keys = args
        slice_spec = None

    try:
        close = _shared_s2["close"]
        open_ = _shared_s2["open"]
        tickers = _shared_s2["tickers"]

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

        # Select KAMA cache for this param set's kama_asset_period
        kama_cache = _shared_s2["kama_caches"].get(params.kama_asset_period)
        # SPY KAMA from kama_spy_period cache
        spy_kama_cache = _shared_s2["kama_caches"].get(params.kama_spy_period, {})
        spy_kama = spy_kama_cache.get("SPY")

        result = run_simulation_s2(
            close, open_, tickers,
            _shared_s2["capital"],
            params=params,
            kama_cache=kama_cache,
            spy_kama_ext=spy_kama,
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


def suggest_params_s2(
    trial: optuna.Trial,
    space: dict[str, dict],
    fixed_params: dict | None = None,
) -> S2StrategyParams:
    """Suggest an S2StrategyParams from an Optuna trial."""
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
    return S2StrategyParams(**kwargs)


def run_optuna_batch_loop_s2(
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
    desc: str = "S2 Optuna trials",
    verbose: bool = True,
) -> pd.DataFrame:
    """Run batch-parallel Optuna ask/tell loop for S2 strategy."""
    if user_attr_keys is None:
        user_attr_keys = metric_keys

    pbar = tqdm(total=n_trials, desc=desc, unit="trial", disable=not verbose)
    trials_done = 0

    while trials_done < n_trials:
        batch_size = min(n_workers, n_trials - trials_done)
        trials = [study.ask() for _ in range(batch_size)]
        params_list = [suggest_params_s2(t, space, fixed_params) for t in trials]

        combo = lambda p: (
            p, max_dd_limit, objective_fn, objective_key,
            param_keys, metric_keys,
        )

        futures = {
            executor.submit(
                evaluate_combo_s2,
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


def select_best_params_s2(
    grid_df: pd.DataFrame,
    objective_col: str,
    param_keys: list[str],
) -> S2StrategyParams | None:
    """Select the best S2 parameter combo from an optimisation grid."""
    valid = grid_df[grid_df[objective_col] > -999.0]
    if valid.empty:
        return None
    best = valid.loc[valid[objective_col].idxmax()]

    _field_types = {
        f.name: f.type
        for f in S2StrategyParams.__dataclass_fields__.values()
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
    return S2StrategyParams(**kwargs)
