"""Maximum profit parameter search for KAMA momentum strategy.

Unlike the sensitivity analysis in optimizer.py (which checks robustness),
this module searches for the parameter combination that maximizes CAGR
over a given period. The drawdown rejection limit is relaxed to 60%.
"""

from __future__ import annotations

import itertools
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
import pandas as pd
import structlog
from tqdm import tqdm

from src.portfolio_sim.config import INITIAL_CAPITAL
from src.portfolio_sim.engine import run_simulation
from src.portfolio_sim.optimizer import precompute_kama_caches
from src.portfolio_sim.params import StrategyParams
from src.portfolio_sim.reporting import compute_metrics

log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Shared data for worker processes (initializer pattern)
# ---------------------------------------------------------------------------
_shared: dict = {}

# ---------------------------------------------------------------------------
# Default grid: 5×6×5×5×2×2×2 = 6,000 combinations
# ---------------------------------------------------------------------------
MAX_PROFIT_GRID: dict[str, list] = {
    "kama_period": [5, 10, 15, 20, 30],
    "lookback_period": [30, 60, 90, 120, 150, 200],
    "top_n": [3, 5, 7, 10, 15],
    "kama_buffer": [0.003, 0.005, 0.008, 0.012, 0.02],
    "use_risk_adjusted": [True, False],
    "enable_regime_filter": [True, False],
    "sizing_mode": ["equal_weight", "risk_parity"],
}

ALL_PARAM_NAMES: list[str] = [
    "kama_period", "lookback_period", "top_n", "kama_buffer",
    "use_risk_adjusted", "enable_regime_filter", "sizing_mode",
    "enable_correlation_filter",
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class MaxProfitResult:
    """Results from a max-profit parameter grid search."""

    universe: str
    grid_results: pd.DataFrame
    default_metrics: dict
    default_params: StrategyParams


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------
def compute_cagr_objective(
    equity: pd.Series, max_dd_limit: float = 0.60,
) -> float:
    """CAGR-maximizing objective with relaxed drawdown limit.

    Returns raw CAGR, or -999.0 for degenerate/rejected curves.
    """
    metrics = compute_metrics(equity)
    if metrics["n_days"] < 60:
        return -999.0
    if metrics["max_drawdown"] > max_dd_limit:
        return -999.0
    cagr = metrics["cagr"]
    if cagr <= 0:
        return -999.0
    return cagr


# ---------------------------------------------------------------------------
# Grid construction (supports booleans and fixed per-universe overrides)
# ---------------------------------------------------------------------------
def build_full_param_grid(
    grid: dict[str, list],
    fixed_params: dict | None = None,
) -> list[StrategyParams]:
    """Expand grid into StrategyParams with optional fixed overrides."""
    fixed_params = fixed_params or {}
    keys = sorted(grid.keys())
    combos = list(itertools.product(*(grid[k] for k in keys)))
    params_list = []
    for vals in combos:
        d = dict(zip(keys, vals))
        d.update(fixed_params)
        params_list.append(StrategyParams(**d))
    return params_list


# ---------------------------------------------------------------------------
# Parallel evaluation helpers
# ---------------------------------------------------------------------------
def _init_max_profit_worker(
    close_prices: pd.DataFrame,
    open_prices: pd.DataFrame,
    tickers: list[str],
    initial_capital: float,
    kama_caches: dict[int, dict[str, pd.Series]],
):
    """Initializer for max-profit grid-search workers."""
    _shared["close"] = close_prices
    _shared["open"] = open_prices
    _shared["tickers"] = tickers
    _shared["capital"] = initial_capital
    _shared["kama_caches"] = kama_caches


def _evaluate_max_profit_combo(args: tuple[StrategyParams, float]) -> dict:
    """Evaluate a single param combo, returning CAGR as the objective."""
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
            obj = compute_cagr_objective(equity, max_dd_limit=max_dd_limit)
            metrics = compute_metrics(equity)
    except (ValueError, KeyError):
        obj = -999.0
        metrics = {}

    return {
        "kama_period": params.kama_period,
        "lookback_period": params.lookback_period,
        "top_n": params.top_n,
        "kama_buffer": params.kama_buffer,
        "use_risk_adjusted": params.use_risk_adjusted,
        "enable_regime_filter": params.enable_regime_filter,
        "sizing_mode": params.sizing_mode,
        "enable_correlation_filter": params.enable_correlation_filter,
        "objective_cagr": obj,
        "total_return": metrics.get("total_return", 0.0),
        "cagr": metrics.get("cagr", 0.0),
        "max_drawdown": metrics.get("max_drawdown", 0.0),
        "sharpe": metrics.get("sharpe", 0.0),
        "calmar": metrics.get("calmar", 0.0),
        "annualized_vol": metrics.get("annualized_vol", 0.0),
        "win_rate": metrics.get("win_rate", 0.0),
    }


# ---------------------------------------------------------------------------
# Main search
# ---------------------------------------------------------------------------
def run_max_profit_search(
    close_prices: pd.DataFrame,
    open_prices: pd.DataFrame,
    tickers: list[str],
    initial_capital: float = INITIAL_CAPITAL,
    universe: str = "sp500",
    default_params: StrategyParams | None = None,
    grid: dict[str, list] | None = None,
    fixed_params: dict | None = None,
    n_workers: int | None = None,
    max_dd_limit: float = 0.60,
) -> MaxProfitResult:
    """Search parameter grid for maximum CAGR.

    1. Expand grid into all combinations (with fixed overrides).
    2. Pre-compute KAMA for all unique periods (parallel).
    3. Evaluate each combination in parallel.
    4. Return results sorted by CAGR.
    """
    default_params = default_params or StrategyParams()
    grid = grid or MAX_PROFIT_GRID
    all_params = build_full_param_grid(grid, fixed_params)
    n_workers = n_workers or max(1, os.cpu_count() - 1)

    log.info(
        "max_profit_start",
        universe=universe,
        n_combos=len(all_params),
        n_workers=n_workers,
    )

    # Pre-compute KAMA for all unique periods
    kama_periods = list(set(p.kama_period for p in all_params))
    kama_caches = precompute_kama_caches(
        close_prices, tickers, kama_periods, n_workers,
    )
    log.info("kama_precomputed", n_periods=len(kama_caches))

    # Run default params to get baseline
    default_kama = kama_caches.get(default_params.kama_period)
    default_result = run_simulation(
        close_prices, open_prices, tickers, initial_capital,
        params=default_params, kama_cache=default_kama,
    )
    default_metrics = compute_metrics(default_result.equity)

    # Evaluate all combos in parallel
    rows: list[dict] = []

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_max_profit_worker,
        initargs=(close_prices, open_prices, tickers, initial_capital, kama_caches),
    ) as executor:
        futures = {
            executor.submit(_evaluate_max_profit_combo, (p, max_dd_limit)): p
            for p in all_params
        }
        bar = tqdm(
            as_completed(futures),
            total=len(all_params),
            desc=f"Grid search ({universe})",
            unit="combo",
        )
        for future in bar:
            try:
                rows.append(future.result())
            except Exception as exc:
                log.warning("combo_failed", error=str(exc))

    grid_df = pd.DataFrame(rows)

    log.info(
        "max_profit_done",
        universe=universe,
        n_valid=int((grid_df["objective_cagr"] > -999.0).sum()),
        n_total=len(grid_df),
    )

    return MaxProfitResult(
        universe=universe,
        grid_results=grid_df,
        default_metrics=default_metrics,
        default_params=default_params,
    )


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------
def format_max_profit_report(
    result: MaxProfitResult,
    top_n: int = 20,
) -> str:
    """Format a human-readable max-profit search report."""
    lines: list[str] = []
    lines.append("=" * 90)
    lines.append(f"MAXIMUM PROFIT SEARCH — {result.universe.upper()} Universe")
    lines.append("=" * 90)

    # Default params baseline
    bp = result.default_params
    dm = result.default_metrics
    lines.append("")
    lines.append("Default Parameters Baseline:")
    lines.append(
        f"  kama_period={bp.kama_period}, lookback_period={bp.lookback_period}, "
        f"top_n={bp.top_n}, kama_buffer={bp.kama_buffer}"
    )
    lines.append(
        f"  risk_adjusted={bp.use_risk_adjusted}, regime_filter={bp.enable_regime_filter}, "
        f"sizing={bp.sizing_mode}"
    )
    lines.append(
        f"  CAGR: {dm['cagr']:.2%}  Total Return: {dm['total_return']:.2%}  "
        f"Max DD: {dm['max_drawdown']:.2%}  Sharpe: {dm['sharpe']:.2f}"
    )

    # Grid summary
    grid_df = result.grid_results
    valid = grid_df[grid_df["objective_cagr"] > -999.0]
    lines.append("")
    lines.append(
        f"Grid: {len(grid_df)} combinations evaluated, "
        f"{len(valid)} valid ({len(valid) / len(grid_df):.0%})"
    )

    if not valid.empty:
        lines.append(
            f"  CAGR range: {valid['cagr'].min():.2%} .. {valid['cagr'].max():.2%}"
        )
        lines.append(f"  Median CAGR: {valid['cagr'].median():.2%}")

    # Top combos
    lines.append("")
    lines.append("-" * 90)
    lines.append(f"Top {top_n} Combinations by CAGR:")
    lines.append("-" * 90)

    if not valid.empty:
        top = valid.nlargest(top_n, "cagr")
        header = (
            f"  {'#':>3} {'kama':>5} {'lbk':>5} {'top_n':>5} {'buf':>7} "
            f"{'rsk_adj':>7} {'regime':>7} {'sizing':>12} "
            f"{'CAGR':>8} {'Return':>9} {'MaxDD':>8} {'Sharpe':>7}"
        )
        lines.append(header)
        lines.append("  " + "-" * 86)
        for rank, (_, row) in enumerate(top.iterrows(), 1):
            lines.append(
                f"  {rank:>3} {int(row['kama_period']):>5} "
                f"{int(row['lookback_period']):>5} "
                f"{int(row['top_n']):>5} {row['kama_buffer']:>7.4f} "
                f"{'Y' if row['use_risk_adjusted'] else 'N':>7} "
                f"{'Y' if row['enable_regime_filter'] else 'N':>7} "
                f"{row['sizing_mode']:>12} "
                f"{row['cagr']:>8.2%} {row['total_return']:>9.2%} "
                f"{row['max_drawdown']:>8.2%} {row['sharpe']:>7.2f}"
            )

    # Best vs default comparison
    if not valid.empty:
        best = valid.nlargest(1, "cagr").iloc[0]
        lines.append("")
        lines.append("-" * 90)
        lines.append("Best vs Default:")
        lines.append("-" * 90)
        lines.append(
            f"  Default CAGR: {dm['cagr']:>8.2%}   |   Best CAGR: {best['cagr']:>8.2%}   "
            f"|   Improvement: {best['cagr'] - dm['cagr']:>+8.2%}"
        )
        lines.append(
            f"  Default Return: {dm['total_return']:>7.2%}  |   Best Return: {best['total_return']:>7.2%}"
        )
        # Compute dollar profit from $10,000
        default_profit = 10_000 * dm["total_return"]
        best_profit = 10_000 * best["total_return"]
        lines.append(
            f"  Default Profit: ${default_profit:>10,.0f}  "
            f"|   Best Profit: ${best_profit:>10,.0f}"
        )

    lines.append("")
    lines.append("=" * 90)
    return "\n".join(lines)
