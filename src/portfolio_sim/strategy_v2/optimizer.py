"""V2 parameter optimisation — Sharpe-based objective with vol-targeting.

Reuses KAMA pre-computation and the generic make_objective factory from
the base optimizer, but targets Sharpe ratio instead of Calmar.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial

import optuna
import pandas as pd
import structlog

from src.portfolio_sim.config import INITIAL_CAPITAL, SPY_TICKER
from src.portfolio_sim.optimizer import (
    SensitivityResult,
    _clamp_to_space,
    _get_kama_periods_from_space,
    precompute_kama_caches,
)
from src.portfolio_sim.reporting import compute_metrics
from src.portfolio_sim.strategy_v2.config import (
    V2_DEFAULT_N_TRIALS,
    V2_MAX_DD_LIMIT,
    V2_PARAM_NAMES,
    V2_SEARCH_SPACE,
)
from src.portfolio_sim.strategy_v2.parallel import (
    run_optuna_batch_loop_v2,
    select_best_params_v2,
)
from src.portfolio_sim.strategy_v2.params import StrategyParamsV2

log = structlog.get_logger()

_V2_METRIC_KEYS = ["cagr", "max_drawdown", "sharpe", "calmar"]


# ---------------------------------------------------------------------------
# V2 objective: raw metric value (no constraints)
# ---------------------------------------------------------------------------
def _raw_metric_objective(
    equity: pd.Series,
    max_dd_limit: float,
    metric: str = "total_return",
) -> float:
    """Raw metric objective — returns metric value with no constraints."""
    return compute_metrics(equity)[metric]


# ---------------------------------------------------------------------------
# Main V2 optimisation
# ---------------------------------------------------------------------------
def run_sensitivity_v2(
    close_prices: pd.DataFrame,
    open_prices: pd.DataFrame,
    tickers: list[str],
    initial_capital: float = INITIAL_CAPITAL,
    base_params: StrategyParamsV2 | None = None,
    space: dict[str, dict] | None = None,
    n_trials: int = V2_DEFAULT_N_TRIALS,
    n_workers: int | None = None,
    max_dd_limit: float = V2_MAX_DD_LIMIT,
    min_n_days: int = 60,
    metric: str = "total_return",
    kama_caches: dict[int, dict[str, pd.Series]] | None = None,
    executor: ProcessPoolExecutor | None = None,
    verbose: bool = True,
) -> SensitivityResult:
    """Run V2 parameter optimisation using Optuna TPE sampler.

    Same structure as v1 run_sensitivity but uses:
      - StrategyParamsV2 (with vol-targeting fields)
      - Sharpe-based objective (instead of Calmar)
      - Tighter max drawdown limit (20%)
    """
    base_params = base_params or StrategyParamsV2()
    space = space or V2_SEARCH_SPACE
    if not n_workers or n_workers < 1:
        n_workers = os.cpu_count()

    log.info(
        "v2_sensitivity_start",
        n_trials=n_trials,
        n_workers=n_workers,
    )

    if kama_caches is None:
        kama_periods = _get_kama_periods_from_space(space)
        kama_caches = precompute_kama_caches(
            close_prices, tickers, kama_periods, n_workers,
        )
        log.info("v2_kama_precomputed", n_periods=len(kama_caches))

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

    from src.portfolio_sim.strategy_v2.parallel import init_eval_worker_v2

    own_executor = executor is None

    if own_executor:
        executor = ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=init_eval_worker_v2,
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

    grid_df = run_optuna_batch_loop_v2(
        study, executor,
        space=space,
        n_trials=n_trials,
        n_workers=n_workers,
        objective_fn=objective_fn,
        max_dd_limit=max_dd_limit,
        objective_key="objective",
        param_keys=V2_PARAM_NAMES,
        metric_keys=_V2_METRIC_KEYS,
        slice_spec=slice_spec,
        desc="V2 sensitivity trials",
        verbose=verbose,
    )

    if own_executor:
        executor.shutdown(wait=True)

    # Compute base params objective
    mask = pd.Series(True, index=grid_df.index)
    for name in V2_PARAM_NAMES:
        if name in grid_df.columns:
            mask = mask & (grid_df[name] == getattr(base_params, name))
    base_row = grid_df[mask]
    if not base_row.empty:
        base_objective = float(base_row.iloc[0]["objective"])
    else:
        base_objective = float("nan")

    log.info(
        "v2_sensitivity_done",
        n_valid=int((grid_df["objective"] > -999.0).sum()),
        n_total=len(grid_df),
    )

    return SensitivityResult(
        grid_results=grid_df,
        base_params=base_params,
        base_objective=base_objective,
    )


def find_best_params_v2(result: SensitivityResult) -> StrategyParamsV2 | None:
    """Extract the best V2 parameter combo from optimisation results."""
    return select_best_params_v2(result.grid_results, "objective", V2_PARAM_NAMES)
