"""Parameter sensitivity analysis for KAMA momentum strategy.

Uses Optuna TPE sampler to efficiently explore the parameter space
and verify that performance is stable across a wide range of values.

Key difference from optimization:
  - Goal is NOT to find "best" parameters (that leads to overfitting).
  - Goal is to confirm chosen defaults sit on a flat performance plateau.
  - If performance varies wildly across the space → strategy is fragile.
  - If performance is stable → parameters are robust, defaults are fine.
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
    SENSITIVITY_SPACE,
    SPY_TICKER,
)
from src.portfolio_sim.indicators import compute_kama_series
from src.portfolio_sim.parallel import (
    _shared,
    suggest_params,
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
]


@dataclass
class SensitivityResult:
    """Results from a sensitivity analysis."""

    grid_results: pd.DataFrame
    """All trial results: columns = param names + objective + metrics."""

    marginal_profiles: dict[str, pd.DataFrame]
    """1D marginal profiles: for each parameter, mean objective per value."""

    robustness_scores: dict[str, float]
    """0-1 score per parameter (1.0 = perfectly flat = robust)."""

    base_params: StrategyParams
    base_objective: float


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------
def compute_objective(
    equity: pd.Series,
    max_dd_limit: float = 0.30,
    min_n_days: int = 60,
) -> float:
    """Total return with drawdown cap and hard rejection.

    Returns:
        Total return (e.g. 0.50 for +50%), or -999.0 when MaxDD
        exceeds *max_dd_limit* or the equity curve is degenerate.
    """
    metrics = compute_metrics(equity)
    if metrics["n_days"] < min_n_days:
        return -999.0
    if metrics["max_drawdown"] > max_dd_limit:
        return -999.0
    total_return = metrics["total_return"]
    if total_return <= 0:
        return -999.0
    return total_return


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


# Sensitivity param/metric keys for evaluate_combo
_SENS_PARAM_KEYS = [
    "kama_period", "lookback_period", "kama_buffer", "top_n",
]
_SENS_METRIC_KEYS = ["cagr", "max_drawdown", "sharpe", "calmar"]


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
    # Only profile params that exist in the grid
    param_names = [n for n in PARAM_NAMES if n in grid_df.columns]
    # Only consider valid results
    valid = grid_df[grid_df["objective"] > -999.0]
    if valid.empty:
        for name in param_names:
            profiles[name] = pd.DataFrame(
                columns=[name, "mean_objective", "std_objective", "count"]
            )
        return profiles

    for name in param_names:
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
    param_names = list(profiles.keys()) if profiles else PARAM_NAMES
    for name in param_names:
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
    space: dict[str, dict] | None = None,
    n_trials: int = DEFAULT_N_TRIALS,
    n_workers: int | None = None,
    max_dd_limit: float = 0.30,
    min_n_days: int = 60,
    kama_caches: dict[int, dict[str, pd.Series]] | None = None,
    executor: ProcessPoolExecutor | None = None,
) -> SensitivityResult:
    """Run parameter sensitivity analysis using Optuna TPE sampler.

    1. Pre-compute KAMA for all possible kama_period values (parallel).
    2. Use Optuna to sample parameter combinations efficiently.
    3. Evaluate each combination in parallel via ProcessPoolExecutor.
    4. Compute 1D marginal profiles (mean objective per parameter value).
    5. Score robustness: flat profile = robust parameter.

    This is NOT optimization — the goal is to verify that performance
    is stable across parameter values, not to find the "best" combo.

    When *kama_caches* is provided, KAMA pre-computation is skipped.
    When *executor* is provided, a shared ProcessPoolExecutor is reused
    instead of creating a new one (workers must hold full-range data;
    per-step slicing is handled via slice_spec in evaluate_combo).
    """
    base_params = base_params or StrategyParams()
    space = space or SENSITIVITY_SPACE
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
    from src.portfolio_sim.parallel import evaluate_combo, init_eval_worker

    # Determine whether we own the executor (and must shut it down).
    own_executor = executor is None

    if own_executor:
        # Use ask/tell API with batch parallelism via ProcessPoolExecutor.
        # This avoids the overhead of Optuna's internal threading (n_jobs)
        # while keeping full process-level parallelism for CPU-bound simulations.
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

    pbar = tqdm(total=n_trials, desc="Sensitivity trials", unit="trial")
    trials_done = 0

    while trials_done < n_trials:
        batch_size = min(n_workers, n_trials - trials_done)
        trials = [study.ask() for _ in range(batch_size)]
        params_list = [suggest_params(t, space) for t in trials]

        combo_base = lambda p: (
            p, max_dd_limit, objective_fn, "objective",
            _SENS_PARAM_KEYS, _SENS_METRIC_KEYS,
        )

        futures = {
            executor.submit(
                evaluate_combo,
                combo_base(p) if slice_spec is None else combo_base(p) + (slice_spec,),
            ): (t, p)
            for t, p in zip(trials, params_list)
        }
        for future in as_completed(futures):
            trial, _ = futures[future]
            result_dict = future.result()
            obj = result_dict["objective"]
            value = obj if obj > -999.0 else float("-inf")
            for key in _SENS_METRIC_KEYS:
                trial.set_user_attr(key, result_dict.get(key, 0.0))
            study.tell(trial, value)
            pbar.update(1)
            trials_done += 1

    pbar.close()
    if own_executor:
        executor.shutdown(wait=True)

    # Extract results into DataFrame
    rows: list[dict] = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            row = dict(trial.params)
            obj_val = trial.value
            row["objective"] = obj_val if obj_val != float("-inf") else -999.0
            row.update(trial.user_attrs)
            rows.append(row)

    grid_df = pd.DataFrame(rows)

    # Compute base params objective — match on all params present in grid
    mask = (
        (grid_df["kama_period"] == base_params.kama_period)
        & (grid_df["lookback_period"] == base_params.lookback_period)
        & (grid_df["kama_buffer"] == base_params.kama_buffer)
        & (grid_df["top_n"] == base_params.top_n)
    )
    base_row = grid_df[mask]
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
        lines.append("  base objective:  N/A (base combo not in trials)")

    # Trials summary
    grid_df = result.grid_results
    valid = grid_df[grid_df["objective"] > -999.0]
    lines.append("")
    lines.append(f"Trials: {len(grid_df)} evaluated, "
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

    profile_names = list(result.marginal_profiles.keys()) or PARAM_NAMES
    for name in profile_names:
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

    score_names = list(result.robustness_scores.keys()) or PARAM_NAMES
    for name in score_names:
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
        header = (
            f"  {'kama':>5} {'lbk':>5} {'buf':>7} {'top_n':>5} "
            f"{'obj':>8} {'cagr':>7} {'maxdd':>7} {'sharpe':>7}"
        )
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


def find_best_params(result: SensitivityResult) -> StrategyParams | None:
    """Extract the best parameter combo from sensitivity results.

    Returns StrategyParams for the combo with highest objective,
    or None if no valid combo was found.
    """
    valid = result.grid_results[result.grid_results["objective"] > -999.0]
    if valid.empty:
        return None
    best = valid.loc[valid["objective"].idxmax()]
    kwargs: dict = {
        "kama_period": int(best["kama_period"]),
        "lookback_period": int(best["lookback_period"]),
        "kama_buffer": float(best["kama_buffer"]),
        "top_n": int(best["top_n"]),
    }
    return StrategyParams(**kwargs)
