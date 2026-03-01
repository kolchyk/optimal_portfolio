"""Parallel evaluation infrastructure for R² Momentum walk-forward optimization.

Uses ProcessPoolExecutor with shared-memory initializer pattern
to avoid repeated pickling of large DataFrames.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import fields

import numpy as np
import optuna
import pandas as pd
from tqdm import tqdm

from src.portfolio_sim.config import SPY_TICKER
from src.portfolio_sim.engine import run_backtest
from src.portfolio_sim.params import R2StrategyParams, R2_PARAM_NAMES
from src.portfolio_sim.reporting import compute_metrics

_METRIC_KEYS = ["cagr", "max_drawdown", "sharpe", "calmar"]

# ---------------------------------------------------------------------------
# Worker shared state
# ---------------------------------------------------------------------------
_r2_shared: dict = {}


def init_r2_worker(
    close_prices: pd.DataFrame,
    open_prices: pd.DataFrame,
    tickers: list[str],
    initial_capital: float,
    kama_caches: dict[int, dict[str, pd.Series]],
) -> None:
    """Initializer for R² evaluation worker processes."""
    _r2_shared["close"] = close_prices
    _r2_shared["open"] = open_prices
    _r2_shared["tickers"] = tickers
    _r2_shared["capital"] = initial_capital
    _r2_shared["kama_caches"] = kama_caches


def evaluate_r2_combo(args: tuple) -> dict:
    """Evaluate a single R² param combo in a worker process."""
    params, metric, slice_spec = args

    close = _r2_shared["close"]
    open_ = _r2_shared["open"]
    tickers = _r2_shared["tickers"]
    capital = _r2_shared["capital"]
    kama_caches = _r2_shared["kama_caches"]

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

    asset_kama = kama_caches.get(params.kama_asset_period, {})
    spy_kama_cache = kama_caches.get(params.kama_spy_period, {})
    spy_kama = spy_kama_cache.get(SPY_TICKER)

    try:
        equity, _, _, _, _ = run_backtest(
            close, open_, tickers,
            initial_capital=capital,
            top_n=params.top_n,
            rebal_period_weeks=params.rebal_period_weeks,
            kama_asset_period=params.kama_asset_period,
            kama_spy_period=params.kama_spy_period,
            kama_buffer=params.kama_buffer,
            r2_lookback=params.r2_lookback,
            gap_threshold=params.gap_threshold,
            atr_period=params.atr_period,
            risk_factor=params.risk_factor,
            kama_cache_ext=asset_kama,
            spy_kama_ext=spy_kama,
        )

        if equity.empty:
            obj = -999.0
            metrics = {}
        else:
            metrics = compute_metrics(equity)
            obj = metrics[metric]
    except (ValueError, KeyError):
        obj = -999.0
        metrics = {}

    row: dict = {}
    for key in R2_PARAM_NAMES:
        row[key] = getattr(params, key)
    row["objective"] = obj
    for key in _METRIC_KEYS:
        row[key] = metrics.get(key, 0.0)
    return row


def suggest_r2_params(
    trial: optuna.Trial,
    space: dict[str, dict],
) -> R2StrategyParams:
    """Suggest R2StrategyParams from an Optuna trial."""
    kwargs = {}
    for name, spec in space.items():
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
    return R2StrategyParams(**kwargs)


# ---------------------------------------------------------------------------
# Optuna batch loop
# ---------------------------------------------------------------------------
def run_r2_optuna_loop(
    study: optuna.Study,
    executor: ProcessPoolExecutor,
    *,
    space: dict[str, dict],
    n_trials: int,
    n_workers: int,
    metric: str = "total_return",
    slice_spec: dict | None = None,
    desc: str = "R2 trials",
    verbose: bool = True,
) -> pd.DataFrame:
    """Batch-parallel Optuna ask/tell loop for R² strategy."""
    pbar = tqdm(total=n_trials, desc=desc, unit="trial", disable=not verbose)
    trials_done = 0

    while trials_done < n_trials:
        batch_size = min(n_workers, n_trials - trials_done)
        trials = [study.ask() for _ in range(batch_size)]
        params_list = [suggest_r2_params(t, space) for t in trials]

        futures = {
            executor.submit(
                evaluate_r2_combo,
                (p, metric, slice_spec),
            ): (t, p)
            for t, p in zip(trials, params_list)
        }
        for future in as_completed(futures):
            trial, _ = futures[future]
            result_dict = future.result()
            obj = result_dict["objective"]
            value = obj if obj > -999.0 else float("-inf")
            for key in _METRIC_KEYS:
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
            row["objective"] = obj_val if obj_val != float("-inf") else -999.0
            row.update(trial.user_attrs)
            rows.append(row)

    return pd.DataFrame(rows)


def select_best_r2_params(grid_df: pd.DataFrame) -> R2StrategyParams | None:
    """Select best R2StrategyParams from optimization grid."""
    valid = grid_df[grid_df["objective"] > -999.0]
    if valid.empty:
        return None
    best = valid.loc[valid["objective"].idxmax()]

    _field_types = {f.name: f.type for f in fields(R2StrategyParams)}
    kwargs = {}
    for key in R2_PARAM_NAMES:
        if key in best.index:
            val = best[key]
            expected = _field_types.get(key)
            if expected == "int" or expected is int:
                val = int(val)
            elif expected == "float" or expected is float:
                val = float(val)
            kwargs[key] = val
    return R2StrategyParams(**kwargs)
