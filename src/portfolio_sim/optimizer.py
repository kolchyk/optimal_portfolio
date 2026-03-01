"""Parameter optimization utilities.

Shared infrastructure for KAMA pre-computation and objective functions.
Used by both the R² Momentum strategy and V2 Vol-Targeted KAMA.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import pandas as pd
from tqdm import tqdm

from src.portfolio_sim.config import INITIAL_CAPITAL, SPY_TICKER
from src.portfolio_sim.indicators import compute_kama_series
from src.portfolio_sim.params import StrategyParams
from src.portfolio_sim.reporting import compute_metrics

METRIC_CHOICES = ("total_return", "cagr", "sharpe", "calmar")

# ---------------------------------------------------------------------------
# Shared data for KAMA worker processes
# ---------------------------------------------------------------------------
_kama_shared: dict = {}


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
# Result dataclass (used by V2 optimizer)
# ---------------------------------------------------------------------------
@dataclass
class SensitivityResult:
    """Results from parameter optimization."""

    grid_results: pd.DataFrame
    base_params: StrategyParams
    base_objective: float


# ---------------------------------------------------------------------------
# Objective functions
# ---------------------------------------------------------------------------
def make_objective(
    metric: str = "total_return",
    max_dd_limit: float = 0.30,
    min_n_days: int = 60,
):
    """Factory that returns an objective function for Optuna trials."""
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
    """Calmar ratio with drawdown cap and hard rejection."""
    return make_objective("calmar", max_dd_limit, min_n_days)(equity)


def r2_objective(
    equity: pd.Series,
    max_dd_limit: float = 0.30,
    min_n_days: int = 20,
) -> float:
    """Calmar-ratio objective for R² Momentum strategy."""
    metrics = compute_metrics(equity)
    if metrics["n_days"] < min_n_days:
        return -999.0
    if metrics["max_drawdown"] > max_dd_limit:
        return -999.0
    cagr = metrics["cagr"]
    if cagr <= 0:
        return -999.0
    calmar = cagr / max(metrics["max_drawdown"], 0.01)
    return calmar


# ---------------------------------------------------------------------------
# KAMA pre-computation (parallel)
# ---------------------------------------------------------------------------
def _init_kama_worker(close_prices: pd.DataFrame):
    """Initializer for KAMA worker processes."""
    _kama_shared["close"] = close_prices


def _compute_single_kama(args: tuple[int, str]) -> tuple[int, str, pd.Series]:
    """Compute KAMA for a single (period, ticker) pair using shared data."""
    period, ticker = args
    series = _kama_shared["close"][ticker].dropna()
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
    """Extract all possible KAMA period values from a search space definition."""
    periods: list[int] = []
    for key in ("kama_period", "kama_spy_period"):
        spec = space.get(key, {})
        if not spec:
            continue
        if spec.get("type") == "categorical":
            periods.extend(spec["choices"])
        else:
            periods.extend(range(
                spec.get("low", 10),
                spec.get("high", 40) + 1,
                spec.get("step", 5),
            ))
    return sorted(set(periods))
