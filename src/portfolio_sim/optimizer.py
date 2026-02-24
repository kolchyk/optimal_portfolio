"""Walk-forward parameter optimization for KAMA momentum strategy.

Ensures no look-ahead bias: parameters are always chosen on in-sample data,
then validated on a strictly future out-of-sample window.
"""

from __future__ import annotations

import itertools
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import structlog
from tqdm import tqdm

from src.portfolio_sim.config import INITIAL_CAPITAL, SPY_TICKER
from src.portfolio_sim.engine import run_simulation
from src.portfolio_sim.indicators import compute_kama_series
from src.portfolio_sim.params import StrategyParams
from src.portfolio_sim.reporting import compute_metrics

log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Shared data for worker processes (initializer pattern avoids repeated
# pickling of large DataFrames — sent once per worker, not once per task).
# ---------------------------------------------------------------------------
_shared: dict = {}

# ---------------------------------------------------------------------------
# Default parameter search grid
# ---------------------------------------------------------------------------
DEFAULT_PARAM_GRID: dict[str, list] = {
    "kama_period": [10, 15, 20, 30, 40],
    "lookback_period": [20, 40, 60, 90, 120],
    "kama_buffer": [0.005, 0.01, 0.015, 0.02, 0.03],
    "top_n": [10, 15, 20, 25, 30],
}


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WalkForwardConfig:
    """Walk-forward optimization settings."""

    min_is_days: int = 756  # 3 years minimum in-sample
    oos_days: int = 252  # 1 year out-of-sample
    step_days: int = 252  # advance OOS window by 1 year
    max_drawdown_limit: float = 0.30  # hard rejection threshold


@dataclass
class FoldResult:
    """Result from a single walk-forward fold."""

    fold_index: int
    is_start: pd.Timestamp
    is_end: pd.Timestamp
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp
    best_params: StrategyParams
    is_objective: float
    oos_equity: pd.Series
    oos_metrics: dict
    all_is_results: dict[StrategyParams, float] = field(default_factory=dict)


@dataclass
class WalkForwardResult:
    """Aggregated results from full walk-forward optimization."""

    fold_results: list[FoldResult]
    concatenated_oos_equity: pd.Series
    oos_metrics: dict
    parameter_stability: pd.DataFrame
    param_sensitivity: dict[str, dict]


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------
def compute_objective(equity: pd.Series, max_dd_limit: float = 0.30) -> float:
    """Calmar ratio with drawdown floor and hard rejection.

    Returns:
        Calmar ratio (CAGR / max(MaxDD, 5%)), or -999.0 when MaxDD
        exceeds *max_dd_limit* or the equity curve is degenerate.
    """
    metrics = compute_metrics(equity)
    if metrics["n_days"] < 60:
        return -999.0
    max_dd = max(metrics["max_drawdown"], 0.05)  # floor at 5%
    if max_dd > max_dd_limit:
        return -999.0
    cagr = metrics["cagr"]
    if cagr <= 0:
        return -999.0
    return cagr / max_dd


# ---------------------------------------------------------------------------
# Fold generation
# ---------------------------------------------------------------------------
def generate_folds(
    dates: pd.DatetimeIndex,
    config: WalkForwardConfig,
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Generate (in_sample_dates, oos_dates) pairs.

    Uses anchored (expanding) in-sample with fixed out-of-sample windows.
    Each OOS window starts strictly after the IS window ends.

    Returns:
        List of (is_dates, oos_dates) tuples. At least 2 folds required.

    Raises:
        ValueError: if fewer than 2 folds can be constructed.
    """
    n = len(dates)
    folds: list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]] = []

    is_end = config.min_is_days  # exclusive index
    while is_end + config.oos_days <= n:
        oos_end = is_end + config.oos_days
        is_dates = dates[:is_end]
        oos_dates = dates[is_end:oos_end]
        folds.append((is_dates, oos_dates))
        is_end += config.step_days

    if len(folds) < 2:
        raise ValueError(
            f"Need at least 2 walk-forward folds. Got {len(folds)} with "
            f"{n} dates (min_is={config.min_is_days}, oos={config.oos_days}). "
            f"Provide more data or reduce window sizes."
        )
    return folds


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
    n_workers = n_workers or max(1, os.cpu_count() - 1)

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


# ---------------------------------------------------------------------------
# Single-fold evaluation helpers
# ---------------------------------------------------------------------------
def _slice_data(
    close_prices: pd.DataFrame,
    open_prices: pd.DataFrame,
    window_dates: pd.DatetimeIndex,
    warmup_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Slice price data for a simulation window, prepending warmup bars."""
    full_index = close_prices.index
    window_start_loc = full_index.get_loc(window_dates[0])
    data_start_loc = max(0, window_start_loc - warmup_days)
    data_end_loc = full_index.get_loc(window_dates[-1]) + 1
    return (
        close_prices.iloc[data_start_loc:data_end_loc],
        open_prices.iloc[data_start_loc:data_end_loc],
    )


def _run_on_window(
    close_prices: pd.DataFrame,
    open_prices: pd.DataFrame,
    tickers: list[str],
    initial_capital: float,
    params: StrategyParams,
    window_dates: pd.DatetimeIndex,
    kama_caches: dict[int, dict[str, pd.Series]],
) -> pd.Series:
    """Run simulation on a specific date window, returning the equity curve.

    Automatically prepends warmup data and trims the result to the window.
    """
    warmup = params.warmup
    close_slice, open_slice = _slice_data(
        close_prices, open_prices, window_dates, warmup
    )
    kama_cache = kama_caches.get(params.kama_period)
    result = run_simulation(
        close_slice, open_slice, tickers, initial_capital,
        params=params, kama_cache=kama_cache,
    )
    equity = result.equity
    # Trim to only the window dates (exclude warmup equity values)
    equity = equity.loc[equity.index.isin(window_dates)]
    return equity


def _init_fold_worker(
    close_prices: pd.DataFrame,
    open_prices: pd.DataFrame,
    tickers: list[str],
    initial_capital: float,
    is_dates: pd.DatetimeIndex,
    kama_caches: dict[int, dict[str, pd.Series]],
):
    """Initializer for fold grid-search workers — stores shared data."""
    _shared["close"] = close_prices
    _shared["open"] = open_prices
    _shared["tickers"] = tickers
    _shared["capital"] = initial_capital
    _shared["is_dates"] = is_dates
    _shared["kama_caches"] = kama_caches


def _evaluate_params_task(args: tuple[StrategyParams, float]) -> tuple[StrategyParams, float]:
    """Evaluate a single param set on the IS window using shared data.

    Lightweight function — only the (params, max_dd_limit) tuple is pickled
    per task; all heavy data lives in _shared (sent once per worker via initializer).
    """
    params, max_dd_limit = args
    try:
        equity = _run_on_window(
            _shared["close"], _shared["open"], _shared["tickers"],
            _shared["capital"], params, _shared["is_dates"], _shared["kama_caches"],
        )
        if equity.empty:
            return params, -999.0
        return params, compute_objective(equity, max_dd_limit=max_dd_limit)
    except (ValueError, KeyError):
        return params, -999.0


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------
def build_param_grid(
    grid: dict[str, list] | None = None,
) -> list[StrategyParams]:
    """Expand a parameter grid dict into a list of StrategyParams."""
    grid = grid or DEFAULT_PARAM_GRID
    keys = sorted(grid.keys())
    combos = list(itertools.product(*(grid[k] for k in keys)))
    return [StrategyParams(**dict(zip(keys, vals))) for vals in combos]


# ---------------------------------------------------------------------------
# Walk-forward main loop
# ---------------------------------------------------------------------------
def run_walk_forward(
    close_prices: pd.DataFrame,
    open_prices: pd.DataFrame,
    tickers: list[str],
    initial_capital: float = INITIAL_CAPITAL,
    param_grid: dict[str, list] | None = None,
    wf_config: WalkForwardConfig | None = None,
    n_workers: int | None = None,
) -> WalkForwardResult:
    """Full walk-forward optimization pipeline.

    For each fold:
      1. Evaluate all param combinations on in-sample data
      2. Select best params by objective
      3. Run best params on out-of-sample data
      4. Record OOS equity curve segment

    After all folds:
      5. Concatenate OOS equity curves
      6. Compute aggregate OOS metrics
      7. Analyze parameter stability across folds
    """
    config = wf_config or WalkForwardConfig()
    grid = param_grid or DEFAULT_PARAM_GRID
    all_params = build_param_grid(grid)
    n_workers = n_workers or max(1, os.cpu_count() - 1)

    log.info(
        "walk_forward_start",
        n_params=len(all_params),
        n_workers=n_workers,
    )

    # Generate folds
    dates = close_prices.index
    folds = generate_folds(dates, config)
    log.info("folds_generated", n_folds=len(folds))

    # Pre-compute KAMA for all unique periods
    kama_periods = list(set(grid.get("kama_period", [20])))
    kama_caches = precompute_kama_caches(close_prices, tickers, kama_periods, n_workers)
    log.info("kama_precomputed", n_periods=len(kama_caches))

    fold_results: list[FoldResult] = []

    fold_bar = tqdm(enumerate(folds), total=len(folds), desc="Walk-forward folds", unit="fold")
    for fold_idx, (is_dates, oos_dates) in fold_bar:
        fold_bar.set_postfix(
            IS=f"{is_dates[0].date()}..{is_dates[-1].date()}",
            OOS=f"{oos_dates[0].date()}..{oos_dates[-1].date()}",
        )
        log.info(
            "fold_start",
            fold=fold_idx + 1,
            total_folds=len(folds),
            is_range=f"{is_dates[0].date()}..{is_dates[-1].date()}",
            oos_range=f"{oos_dates[0].date()}..{oos_dates[-1].date()}",
        )

        # --- Evaluate all params on IS (parallel, initializer pattern) ---
        is_results: dict[StrategyParams, float] = {}

        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_fold_worker,
            initargs=(close_prices, open_prices, tickers,
                      initial_capital, is_dates, kama_caches),
        ) as executor:
            futures = {
                executor.submit(
                    _evaluate_params_task, (p, config.max_drawdown_limit),
                ): p
                for p in all_params
            }
            grid_bar = tqdm(
                as_completed(futures),
                total=len(all_params),
                desc=f"  Fold {fold_idx + 1}/{len(folds)} grid search",
                unit="combo",
                leave=False,
            )
            for future in grid_bar:
                p, obj = future.result()
                is_results[p] = obj

        # --- Pick best IS params ---
        best_params = max(is_results, key=is_results.get)  # type: ignore[arg-type]
        best_is_obj = is_results[best_params]

        log.info(
            "fold_best_is",
            fold=fold_idx + 1,
            objective=round(best_is_obj, 4),
            params=str(best_params),
        )

        # --- Run best params on OOS ---
        oos_equity = _run_on_window(
            close_prices, open_prices, tickers, initial_capital,
            best_params, oos_dates, kama_caches,
        )
        oos_metrics = compute_metrics(oos_equity)

        fold_results.append(
            FoldResult(
                fold_index=fold_idx,
                is_start=is_dates[0],
                is_end=is_dates[-1],
                oos_start=oos_dates[0],
                oos_end=oos_dates[-1],
                best_params=best_params,
                is_objective=best_is_obj,
                oos_equity=oos_equity,
                oos_metrics=oos_metrics,
                all_is_results=is_results,
            )
        )

        log.info(
            "fold_oos_done",
            fold=fold_idx + 1,
            oos_cagr=round(oos_metrics.get("cagr", 0.0), 4),
            oos_maxdd=round(oos_metrics.get("max_drawdown", 0.0), 4),
        )

    # --- Aggregate results ---
    concat_equity = concatenate_oos_equity(fold_results)
    agg_metrics = compute_metrics(concat_equity)
    stability = analyze_parameter_stability(fold_results)
    sensitivity = analyze_param_sensitivity(fold_results)

    log.info(
        "walk_forward_done",
        oos_cagr=round(agg_metrics.get("cagr", 0.0), 4),
        oos_sharpe=round(agg_metrics.get("sharpe", 0.0), 4),
        oos_maxdd=round(agg_metrics.get("max_drawdown", 0.0), 4),
    )

    return WalkForwardResult(
        fold_results=fold_results,
        concatenated_oos_equity=concat_equity,
        oos_metrics=agg_metrics,
        parameter_stability=stability,
        param_sensitivity=sensitivity,
    )


# ---------------------------------------------------------------------------
# OOS equity curve concatenation
# ---------------------------------------------------------------------------
def concatenate_oos_equity(fold_results: list[FoldResult]) -> pd.Series:
    """Chain OOS equity curves, scaling each to continue from the previous.

    The first fold starts at its original initial capital. Each subsequent
    fold is rescaled so its starting value equals the ending value of the
    previous fold.
    """
    if not fold_results:
        return pd.Series(dtype=float)

    segments: list[pd.Series] = []
    running_capital = fold_results[0].oos_equity.iloc[0]

    for fr in fold_results:
        oos_eq = fr.oos_equity
        if oos_eq.empty:
            continue
        scale = running_capital / oos_eq.iloc[0]
        scaled = oos_eq * scale
        segments.append(scaled)
        running_capital = scaled.iloc[-1]

    return pd.concat(segments)


# ---------------------------------------------------------------------------
# Anti-overfitting analysis
# ---------------------------------------------------------------------------
def analyze_parameter_stability(
    fold_results: list[FoldResult],
) -> pd.DataFrame:
    """Build a table of best parameters per fold for stability checking.

    A parameter that varies wildly across folds is likely overfit.
    """
    rows = []
    for fr in fold_results:
        bp = fr.best_params
        rows.append(
            {
                "fold": fr.fold_index + 1,
                "kama_period": bp.kama_period,
                "lookback_period": bp.lookback_period,
                "kama_buffer": bp.kama_buffer,
                "top_n": bp.top_n,
                "is_objective": round(fr.is_objective, 4),
                "oos_cagr": round(fr.oos_metrics.get("cagr", 0.0), 4),
                "oos_maxdd": round(fr.oos_metrics.get("max_drawdown", 0.0), 4),
                "oos_calmar": round(fr.oos_metrics.get("calmar", 0.0), 4),
            }
        )
    return pd.DataFrame(rows)


def analyze_param_sensitivity(
    fold_results: list[FoldResult],
) -> dict[str, dict]:
    """For each parameter, compute how the IS objective changes when that
    single parameter is varied while others are held at the optimum.

    A flat profile indicates robustness; a sharp peak indicates overfitting.

    Returns:
        {param_name: {fold_idx: {value: objective}}}
    """
    param_names = ["kama_period", "lookback_period", "kama_buffer", "top_n"]
    result: dict[str, dict] = {name: {} for name in param_names}

    for fr in fold_results:
        bp = fr.best_params
        is_results = fr.all_is_results
        if not is_results:
            continue

        for param_name in param_names:
            best_val = getattr(bp, param_name)
            profile: dict = {}

            for p, obj in is_results.items():
                # Check if all OTHER params match the best
                other_match = all(
                    getattr(p, n) == getattr(bp, n)
                    for n in param_names
                    if n != param_name
                )
                if other_match:
                    profile[getattr(p, param_name)] = obj

            result[param_name][fr.fold_index] = {
                "best_value": best_val,
                "profile": dict(sorted(profile.items())),
            }

    return result


def compute_stability_scores(
    stability_df: pd.DataFrame,
    param_grid: dict[str, list] | None = None,
) -> dict[str, float]:
    """Compute a stability score (0-1) for each parameter.

    Score = 1 means the parameter never changed across folds.
    Score < 0.5 is a warning (parameter varies over >50% of grid range).
    """
    grid = param_grid or DEFAULT_PARAM_GRID
    scores: dict[str, float] = {}

    for param_name in ["kama_period", "lookback_period", "kama_buffer", "top_n"]:
        if param_name not in stability_df.columns:
            continue
        values = stability_df[param_name].values
        grid_values = sorted(grid.get(param_name, []))
        if len(grid_values) <= 1:
            scores[param_name] = 1.0
            continue
        grid_range = grid_values[-1] - grid_values[0]
        if grid_range == 0:
            scores[param_name] = 1.0
            continue
        value_range = values.max() - values.min()
        scores[param_name] = 1.0 - (value_range / grid_range)

    return scores


def compute_is_oos_degradation(fold_results: list[FoldResult]) -> list[float]:
    """Compute IS vs OOS degradation ratio for each fold.

    degradation = 1 - (oos_calmar / is_objective)
    Values > 0.5 suggest overfitting.
    """
    degradations = []
    for fr in fold_results:
        oos_calmar = fr.oos_metrics.get("calmar", 0.0)
        if fr.is_objective > 0:
            degradations.append(1.0 - (oos_calmar / fr.is_objective))
        else:
            degradations.append(1.0)
    return degradations


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------
def format_walk_forward_report(result: WalkForwardResult) -> str:
    """Format a human-readable walk-forward optimization report."""
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("WALK-FORWARD OPTIMIZATION REPORT")
    lines.append("=" * 70)

    # Aggregate OOS metrics
    m = result.oos_metrics
    lines.append("")
    lines.append("Out-of-Sample Aggregate Performance:")
    lines.append(f"  CAGR:          {m.get('cagr', 0):.2%}")
    lines.append(f"  Max Drawdown:  {m.get('max_drawdown', 0):.2%}")
    lines.append(f"  Sharpe:        {m.get('sharpe', 0):.2f}")
    lines.append(f"  Calmar:        {m.get('calmar', 0):.2f}")
    lines.append(f"  Ann. Vol:      {m.get('annualized_vol', 0):.2%}")

    # Per-fold summary
    lines.append("")
    lines.append("-" * 70)
    lines.append("Per-Fold Results:")
    lines.append("-" * 70)
    stab = result.parameter_stability
    lines.append(stab.to_string(index=False))

    # Stability scores
    lines.append("")
    lines.append("-" * 70)
    lines.append("Parameter Stability (1.0 = perfectly stable):")
    lines.append("-" * 70)
    scores = compute_stability_scores(stab)
    for name, score in scores.items():
        flag = " WARNING: unstable" if score < 0.5 else ""
        lines.append(f"  {name:20s}: {score:.2f}{flag}")

    # IS vs OOS degradation
    degradations = compute_is_oos_degradation(result.fold_results)
    avg_deg = np.mean(degradations) if degradations else 0.0
    lines.append("")
    lines.append("-" * 70)
    lines.append(f"IS→OOS Degradation (avg): {avg_deg:.2%}")
    if avg_deg > 0.5:
        lines.append("  WARNING: High degradation suggests overfitting!")
    lines.append("-" * 70)

    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)
