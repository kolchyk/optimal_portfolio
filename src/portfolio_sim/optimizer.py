"""Parameter sensitivity analysis for KAMA momentum strategy.

Replaces walk-forward optimization with a simpler robustness check:
run a coarse parameter grid on the full dataset and verify that
performance is stable across a wide range of parameter values.

Key difference from optimization:
  - Goal is NOT to find "best" parameters (that leads to overfitting).
  - Goal is to confirm chosen defaults sit on a flat performance plateau.
  - If performance varies wildly across the grid → strategy is fragile.
  - If performance is stable → parameters are robust, defaults are fine.
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
# Default sensitivity grid (5 values per parameter = 625 combinations)
# ---------------------------------------------------------------------------
SENSITIVITY_GRID: dict[str, list] = {
    "kama_period": [10, 15, 20, 30, 40],
    "lookback_period": [20, 40, 60, 90, 120],
    "kama_buffer": [0.005, 0.01, 0.015, 0.02, 0.03],
    "top_n": [10, 15, 20, 25, 30],
}

PARAM_NAMES: list[str] = ["kama_period", "lookback_period", "kama_buffer", "top_n"]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class SensitivityResult:
    """Results from a full-grid sensitivity analysis."""

    grid_results: pd.DataFrame
    """All combo results: columns = param names + objective + metrics."""

    marginal_profiles: dict[str, pd.DataFrame]
    """1D marginal profiles: for each parameter, mean objective per value."""

    robustness_scores: dict[str, float]
    """0-1 score per parameter (1.0 = perfectly flat = robust)."""

    base_params: StrategyParams
    base_objective: float


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
# Grid construction
# ---------------------------------------------------------------------------
def build_param_grid(
    grid: dict[str, list] | None = None,
) -> list[StrategyParams]:
    """Expand a parameter grid dict into a list of StrategyParams."""
    grid = grid or SENSITIVITY_GRID
    keys = sorted(grid.keys())
    combos = list(itertools.product(*(grid[k] for k in keys)))
    return [StrategyParams(**dict(zip(keys, vals))) for vals in combos]


# ---------------------------------------------------------------------------
# Parallel evaluation helpers
# ---------------------------------------------------------------------------
def _init_sensitivity_worker(
    close_prices: pd.DataFrame,
    open_prices: pd.DataFrame,
    tickers: list[str],
    initial_capital: float,
    kama_caches: dict[int, dict[str, pd.Series]],
):
    """Initializer for sensitivity grid-search workers."""
    _shared["close"] = close_prices
    _shared["open"] = open_prices
    _shared["tickers"] = tickers
    _shared["capital"] = initial_capital
    _shared["kama_caches"] = kama_caches


def _evaluate_combo(args: tuple[StrategyParams, float]) -> dict:
    """Evaluate a single param combo on the full dataset.

    Returns a dict with parameter values, objective, and key metrics.
    """
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
            obj = compute_objective(equity, max_dd_limit=max_dd_limit)
            metrics = compute_metrics(equity)
    except (ValueError, KeyError):
        obj = -999.0
        metrics = {}

    return {
        "kama_period": params.kama_period,
        "lookback_period": params.lookback_period,
        "kama_buffer": params.kama_buffer,
        "top_n": params.top_n,
        "objective": obj,
        "cagr": metrics.get("cagr", 0.0),
        "max_drawdown": metrics.get("max_drawdown", 0.0),
        "sharpe": metrics.get("sharpe", 0.0),
        "calmar": metrics.get("calmar", 0.0),
    }


# ---------------------------------------------------------------------------
# Marginal profiles & robustness scores
# ---------------------------------------------------------------------------
def compute_marginal_profiles(
    grid_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """For each parameter, compute the mean objective across all other params.

    Returns:
        {param_name: DataFrame with columns [param_name, "mean_objective",
         "std_objective", "count"]}
    """
    profiles: dict[str, pd.DataFrame] = {}
    # Only consider valid results
    valid = grid_df[grid_df["objective"] > -999.0]
    if valid.empty:
        for name in PARAM_NAMES:
            profiles[name] = pd.DataFrame(
                columns=[name, "mean_objective", "std_objective", "count"]
            )
        return profiles

    for name in PARAM_NAMES:
        agg = (
            valid.groupby(name)["objective"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        agg.columns = [name, "mean_objective", "std_objective", "count"]
        profiles[name] = agg

    return profiles


def compute_robustness_scores(
    profiles: dict[str, pd.DataFrame],
) -> dict[str, float]:
    """Score each parameter's robustness on a 0-1 scale.

    Score = 1 - coefficient_of_variation(mean_objectives), clamped to [0, 1].
    A score of 1.0 means the mean objective is perfectly flat across all values
    of that parameter (= robust). Low scores indicate sensitivity.
    """
    scores: dict[str, float] = {}
    for name in PARAM_NAMES:
        profile = profiles.get(name)
        if profile is None or profile.empty:
            scores[name] = 0.0
            continue
        means = profile["mean_objective"].values
        if len(means) < 2:
            scores[name] = 1.0
            continue
        avg = np.mean(means)
        if avg <= 0:
            scores[name] = 0.0
            continue
        cv = np.std(means) / avg
        scores[name] = float(np.clip(1.0 - cv, 0.0, 1.0))

    return scores


# ---------------------------------------------------------------------------
# Main sensitivity analysis
# ---------------------------------------------------------------------------
def run_sensitivity(
    close_prices: pd.DataFrame,
    open_prices: pd.DataFrame,
    tickers: list[str],
    initial_capital: float = INITIAL_CAPITAL,
    base_params: StrategyParams | None = None,
    grid: dict[str, list] | None = None,
    n_workers: int | None = None,
    max_dd_limit: float = 0.30,
) -> SensitivityResult:
    """Run parameter sensitivity analysis on the full dataset.

    1. Expand grid into all combinations.
    2. Pre-compute KAMA for all unique periods (parallel).
    3. Evaluate each combination in parallel on the full date range.
    4. Compute 1D marginal profiles (mean objective per parameter value).
    5. Score robustness: flat profile = robust parameter.

    This is NOT optimization — the goal is to verify that performance
    is stable across parameter values, not to find the "best" combo.
    """
    base_params = base_params or StrategyParams()
    grid = grid or SENSITIVITY_GRID
    all_params = build_param_grid(grid)
    n_workers = n_workers or max(1, os.cpu_count() - 1)

    log.info(
        "sensitivity_start",
        n_combos=len(all_params),
        n_workers=n_workers,
    )

    # Pre-compute KAMA for all unique periods
    kama_periods = list(set(grid.get("kama_period", [base_params.kama_period])))
    kama_caches = precompute_kama_caches(close_prices, tickers, kama_periods, n_workers)
    log.info("kama_precomputed", n_periods=len(kama_caches))

    # Evaluate all combos in parallel
    rows: list[dict] = []

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_sensitivity_worker,
        initargs=(close_prices, open_prices, tickers, initial_capital, kama_caches),
    ) as executor:
        futures = {
            executor.submit(_evaluate_combo, (p, max_dd_limit)): p
            for p in all_params
        }
        bar = tqdm(
            as_completed(futures),
            total=len(all_params),
            desc="Sensitivity grid",
            unit="combo",
        )
        for future in bar:
            rows.append(future.result())

    grid_df = pd.DataFrame(rows)

    # Compute base params objective
    base_row = grid_df[
        (grid_df["kama_period"] == base_params.kama_period)
        & (grid_df["lookback_period"] == base_params.lookback_period)
        & (grid_df["kama_buffer"] == base_params.kama_buffer)
        & (grid_df["top_n"] == base_params.top_n)
    ]
    if not base_row.empty:
        base_objective = float(base_row.iloc[0]["objective"])
    else:
        base_objective = float("nan")

    profiles = compute_marginal_profiles(grid_df)
    scores = compute_robustness_scores(profiles)

    log.info(
        "sensitivity_done",
        n_valid=int((grid_df["objective"] > -999.0).sum()),
        n_total=len(grid_df),
        robustness=scores,
    )

    return SensitivityResult(
        grid_results=grid_df,
        marginal_profiles=profiles,
        robustness_scores=scores,
        base_params=base_params,
        base_objective=base_objective,
    )


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------
def format_sensitivity_report(result: SensitivityResult) -> str:
    """Format a human-readable sensitivity analysis report."""
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("PARAMETER SENSITIVITY ANALYSIS REPORT")
    lines.append("=" * 70)

    # Base params info
    bp = result.base_params
    lines.append("")
    lines.append("Base Parameters:")
    lines.append(f"  kama_period:     {bp.kama_period}")
    lines.append(f"  lookback_period: {bp.lookback_period}")
    lines.append(f"  kama_buffer:     {bp.kama_buffer}")
    lines.append(f"  top_n:           {bp.top_n}")
    if not np.isnan(result.base_objective):
        lines.append(f"  base objective:  {result.base_objective:.4f}")
    else:
        lines.append("  base objective:  N/A (base combo not in grid)")

    # Grid summary
    grid_df = result.grid_results
    valid = grid_df[grid_df["objective"] > -999.0]
    lines.append("")
    lines.append(f"Grid: {len(grid_df)} combinations evaluated, "
                 f"{len(valid)} valid ({len(valid)/len(grid_df):.0%})")

    if not valid.empty:
        lines.append(f"  Objective range: {valid['objective'].min():.4f} .. "
                     f"{valid['objective'].max():.4f}")
        lines.append(f"  Median objective: {valid['objective'].median():.4f}")

    # Marginal profiles
    lines.append("")
    lines.append("-" * 70)
    lines.append("Marginal Profiles (mean objective per parameter value):")
    lines.append("-" * 70)

    for name in PARAM_NAMES:
        profile = result.marginal_profiles.get(name)
        if profile is None or profile.empty:
            lines.append(f"\n  {name}: no valid data")
            continue

        lines.append(f"\n  {name}:")
        for _, row in profile.iterrows():
            val = row[name]
            mean_obj = row["mean_objective"]
            std_obj = row["std_objective"]
            count = int(row["count"])
            if isinstance(val, float):
                lines.append(f"    {val:8.4f} → {mean_obj:.4f} (±{std_obj:.4f}, n={count})")
            else:
                lines.append(f"    {val:8} → {mean_obj:.4f} (±{std_obj:.4f}, n={count})")

    # Robustness scores
    lines.append("")
    lines.append("-" * 70)
    lines.append("Robustness Scores (1.0 = perfectly flat = robust):")
    lines.append("-" * 70)

    for name in PARAM_NAMES:
        score = result.robustness_scores.get(name, 0.0)
        if score >= 0.8:
            verdict = "ROBUST"
        elif score >= 0.5:
            verdict = "MODERATE"
        else:
            verdict = "SENSITIVE — consider narrowing this parameter"
        lines.append(f"  {name:20s}: {score:.2f}  {verdict}")

    # Top combos
    lines.append("")
    lines.append("-" * 70)
    lines.append("Top 10 Combinations by Objective:")
    lines.append("-" * 70)

    if not valid.empty:
        top = valid.nlargest(10, "objective")
        header = f"  {'kama':>5} {'lbk':>5} {'buf':>7} {'top_n':>5} {'obj':>8} {'cagr':>7} {'maxdd':>7} {'sharpe':>7}"
        lines.append(header)
        for _, row in top.iterrows():
            lines.append(
                f"  {int(row['kama_period']):>5} {int(row['lookback_period']):>5} "
                f"{row['kama_buffer']:>7.4f} {int(row['top_n']):>5} "
                f"{row['objective']:>8.4f} {row['cagr']:>7.2%} "
                f"{row['max_drawdown']:>7.2%} {row['sharpe']:>7.2f}"
            )

    # Overall verdict
    lines.append("")
    lines.append("=" * 70)
    avg_score = np.mean(list(result.robustness_scores.values()))
    if avg_score >= 0.8:
        lines.append("VERDICT: Parameters are ROBUST. Current defaults are well-chosen.")
    elif avg_score >= 0.5:
        lines.append("VERDICT: Parameters are MODERATELY robust. Review sensitive params.")
    else:
        lines.append("VERDICT: Parameters are FRAGILE. Strategy may be overfit to specific values.")
    lines.append("=" * 70)

    return "\n".join(lines)
