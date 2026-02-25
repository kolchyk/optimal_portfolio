## Tree for src
```
├── __init__.py
├── __pycache__/
│   └── __init__.cpython-314.pyc
└── portfolio_sim/
    ├── max_profit.py
    ├── cli_utils.py
    ├── params.py
    ├── config.py
    ├── models.py
    ├── __init__.py
    ├── indicators.py
    ├── __pycache__/
    │   ├── config.cpython-314.pyc
    │   ├── optimization.cpython-314.pyc
    │   ├── data.cpython-314.pyc
    │   ├── cli_utils.cpython-314.pyc
    │   ├── indicators._kama_recurrent_loop-11.py314.nbi
    │   ├── reporting.cpython-314.pyc
    │   ├── indicators.cpython-314.pyc
    │   ├── optimizer.cpython-314.pyc
    │   ├── params.cpython-314.pyc
    │   ├── cli.cpython-314.pyc
    │   ├── models.cpython-314.pyc
    │   ├── parallel.cpython-314.pyc
    │   ├── __init__.cpython-314.pyc
    │   ├── __main__.cpython-314.pyc
    │   ├── alpha.cpython-314.pyc
    │   ├── indicators._kama_recurrent_loop-11.py314.1.nbc
    │   ├── max_profit.cpython-314.pyc
    │   ├── engine.cpython-314.pyc
    │   └── walk_forward.cpython-314.pyc
    ├── engine.py
    ├── cli.py
    ├── alpha.py
    ├── optimizer.py
    ├── commands/
    │   ├── max_profit.py
    │   ├── optimize.py
    │   ├── __init__.py
    │   ├── __pycache__/
    │   │   ├── simulate.cpython-314.pyc
    │   │   ├── backtest.cpython-314.pyc
    │   │   ├── optimize.cpython-314.pyc
    │   │   ├── __init__.cpython-314.pyc
    │   │   ├── max_profit.cpython-314.pyc
    │   │   └── walk_forward.cpython-314.pyc
    │   └── walk_forward.py
    ├── walk_forward.py
    ├── parallel.py
    ├── __main__.py
    ├── data.py
    └── reporting.py
```

## File: __pycache__/__init__.cpython-314.pyc
```
Error reading src/__pycache__/__init__.cpython-314.pyc: 'utf-8' codec can't decode byte 0xe3 in position 16: invalid continuation byte
```
## File: portfolio_sim/max_profit.py
```python
"""Maximum profit parameter search for KAMA momentum strategy.

Unlike the sensitivity analysis in optimizer.py (which checks robustness),
this module searches for the parameter combination that maximizes CAGR
over a given period using Optuna's TPE sampler. The drawdown rejection
limit is relaxed to 60%.

Also provides a multi-objective Pareto search that simultaneously optimizes
CAGR (maximize) and MaxDD (minimize) using NSGA-II, producing a Pareto
front of non-dominated solutions.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import optuna
import pandas as pd
import structlog
from tqdm import tqdm

from src.portfolio_sim.config import INITIAL_CAPITAL
from src.portfolio_sim.engine import run_simulation
from src.portfolio_sim.optimizer import (
    _get_kama_periods_from_space,
    precompute_kama_caches,
)
from src.portfolio_sim.parallel import (
    evaluate_combo,
    init_eval_worker,
    suggest_params,
)
from src.portfolio_sim.params import StrategyParams
from src.portfolio_sim.reporting import compute_metrics

log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Default search space for max profit search
# ---------------------------------------------------------------------------
MAX_PROFIT_SPACE: dict[str, dict] = {
    "kama_period": {"type": "categorical", "choices": [10, 15, 20, 30]},
    "lookback_period": {"type": "int", "low": 20, "high": 100, "step": 20},
    "top_n": {"type": "int", "low": 5, "high": 30, "step": 5},
    "kama_buffer": {"type": "float", "low": 0.005, "high": 0.03, "step": 0.005},
    "enable_regime_filter": {"type": "categorical", "choices": [True, False]},
}

DEFAULT_MAX_PROFIT_TRIALS: int = 50

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
    """Results from a max-profit parameter search."""

    universe: str
    grid_results: pd.DataFrame
    default_metrics: dict
    default_params: StrategyParams
    pareto_front: pd.DataFrame | None = None  # populated only for Pareto search


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


# Max-profit param/metric keys for evaluate_combo
_MP_PARAM_KEYS = [
    "kama_period", "lookback_period", "top_n", "kama_buffer",
    "use_risk_adjusted", "enable_regime_filter", "sizing_mode",
    "enable_correlation_filter",
]
_MP_METRIC_KEYS = [
    "total_return", "cagr", "max_drawdown", "sharpe", "calmar",
    "annualized_vol", "win_rate",
]
_MP_USER_ATTR_KEYS = _MP_METRIC_KEYS + [
    "use_risk_adjusted", "enable_regime_filter", "sizing_mode",
    "enable_correlation_filter",
]


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
    space: dict[str, dict] | None = None,
    fixed_params: dict | None = None,
    n_trials: int = DEFAULT_MAX_PROFIT_TRIALS,
    n_workers: int | None = None,
    max_dd_limit: float = 0.60,
) -> MaxProfitResult:
    """Search parameter space for maximum CAGR using Optuna TPE.

    1. Pre-compute KAMA for all possible kama_period values (parallel).
    2. Use Optuna TPE sampler to explore the parameter space efficiently.
    3. Evaluate each combination in parallel via ProcessPoolExecutor.
    4. Return results sorted by CAGR.
    """
    default_params = default_params or StrategyParams()
    space = space or MAX_PROFIT_SPACE
    if not n_workers or n_workers < 1:
        n_workers = max(1, os.cpu_count() - 1)

    log.info(
        "max_profit_start",
        universe=universe,
        n_trials=n_trials,
        n_workers=n_workers,
    )

    # Pre-compute KAMA for all possible kama_period values
    kama_periods = _get_kama_periods_from_space(space)
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

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    # Use ask/tell API with batch parallelism via ProcessPoolExecutor.
    executor = ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=init_eval_worker,
        initargs=(close_prices, open_prices, tickers, initial_capital, kama_caches),
    )

    pbar = tqdm(total=n_trials, desc=f"Optuna search ({universe})", unit="trial")
    trials_done = 0

    while trials_done < n_trials:
        batch_size = min(n_workers, n_trials - trials_done)
        trials = [study.ask() for _ in range(batch_size)]
        params_list = [suggest_params(t, space, fixed_params) for t in trials]

        futures = {
            executor.submit(
                evaluate_combo,
                (p, max_dd_limit, compute_cagr_objective, "objective_cagr",
                 _MP_PARAM_KEYS, _MP_METRIC_KEYS),
            ): (t, p)
            for t, p in zip(trials, params_list)
        }
        for future in as_completed(futures):
            trial, _ = futures[future]
            result_dict = future.result()
            obj = result_dict["objective_cagr"]
            value = obj if obj > -999.0 else float("-inf")
            for key in _MP_USER_ATTR_KEYS:
                trial.set_user_attr(key, result_dict.get(key, 0.0))
            study.tell(trial, value)
            pbar.update(1)
            trials_done += 1

    pbar.close()
    executor.shutdown(wait=True)

    # Extract results into DataFrame
    rows: list[dict] = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            row = dict(trial.params)
            obj_val = trial.value
            row["objective_cagr"] = obj_val if obj_val != float("-inf") else -999.0
            row.update(trial.user_attrs)
            rows.append(row)

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

    # Trials summary
    grid_df = result.grid_results
    valid = grid_df[grid_df["objective_cagr"] > -999.0]
    lines.append("")
    lines.append(
        f"Trials: {len(grid_df)} evaluated, "
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


# ---------------------------------------------------------------------------
# Multi-objective Pareto search (CAGR ↑, MaxDD ↓)
# ---------------------------------------------------------------------------
def _compute_pareto_objectives(
    equity: pd.Series, max_dd_limit: float = 0.60,
) -> tuple[float, float]:
    """Return (CAGR, MaxDD) for multi-objective optimization.

    Returns (−inf, +inf) for degenerate/rejected curves.
    """
    metrics = compute_metrics(equity)
    if metrics["n_days"] < 60 or metrics["cagr"] <= 0:
        return float("-inf"), float("inf")
    return metrics["cagr"], metrics["max_drawdown"]


def run_max_profit_pareto(
    close_prices: pd.DataFrame,
    open_prices: pd.DataFrame,
    tickers: list[str],
    initial_capital: float = INITIAL_CAPITAL,
    universe: str = "etf",
    default_params: StrategyParams | None = None,
    space: dict[str, dict] | None = None,
    fixed_params: dict | None = None,
    n_trials: int = DEFAULT_MAX_PROFIT_TRIALS,
    n_workers: int | None = None,
) -> MaxProfitResult:
    """Multi-objective search: maximize CAGR, minimize MaxDD (NSGA-II).

    Returns a MaxProfitResult with the ``pareto_front`` field populated.
    The ``grid_results`` DataFrame contains all trials, and the Pareto
    front is the subset of non-dominated solutions.
    """
    default_params = default_params or StrategyParams()
    space = space or MAX_PROFIT_SPACE
    if not n_workers or n_workers < 1:
        n_workers = max(1, os.cpu_count() - 1)

    log.info(
        "pareto_start",
        universe=universe,
        n_trials=n_trials,
        n_workers=n_workers,
    )

    # Pre-compute KAMA
    kama_periods = _get_kama_periods_from_space(space)
    kama_caches = precompute_kama_caches(
        close_prices, tickers, kama_periods, n_workers,
    )

    # Default baseline
    default_kama = kama_caches.get(default_params.kama_period)
    default_result = run_simulation(
        close_prices, open_prices, tickers, initial_capital,
        params=default_params, kama_cache=default_kama,
    )
    default_metrics = compute_metrics(default_result.equity)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        directions=["maximize", "minimize"],  # CAGR ↑, MaxDD ↓
        sampler=optuna.samplers.NSGAIISampler(seed=42),
    )

    executor = ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=init_eval_worker,
        initargs=(close_prices, open_prices, tickers, initial_capital, kama_caches),
    )

    pbar = tqdm(total=n_trials, desc=f"Pareto search ({universe})", unit="trial")
    trials_done = 0

    while trials_done < n_trials:
        batch_size = min(n_workers, n_trials - trials_done)
        trials = [study.ask() for _ in range(batch_size)]
        params_list = [suggest_params(t, space, fixed_params) for t in trials]

        # Reuse evaluate_combo which returns dict with cagr and max_drawdown
        futures = {
            executor.submit(
                evaluate_combo,
                (p, 1.0, compute_cagr_objective, "objective_cagr",
                 _MP_PARAM_KEYS, _MP_METRIC_KEYS),
            ): (t, p)
            for t, p in zip(trials, params_list)
        }
        for future in as_completed(futures):
            trial, _ = futures[future]
            result_dict = future.result()
            cagr = result_dict.get("cagr", 0.0)
            maxdd = result_dict.get("max_drawdown", 0.0)

            if cagr <= 0 or result_dict.get("objective_cagr", -999.0) <= -999.0:
                values = [float("-inf"), float("inf")]
            else:
                values = [cagr, maxdd]

            for key in _MP_USER_ATTR_KEYS:
                trial.set_user_attr(key, result_dict.get(key, 0.0))
            study.tell(trial, values)
            pbar.update(1)
            trials_done += 1

    pbar.close()
    executor.shutdown(wait=True)

    # Extract all results
    rows: list[dict] = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            row = dict(trial.params)
            if trial.values and len(trial.values) == 2:
                row["objective_cagr"] = trial.values[0] if trial.values[0] != float("-inf") else -999.0
                row["objective_maxdd"] = trial.values[1] if trial.values[1] != float("inf") else 999.0
            else:
                row["objective_cagr"] = -999.0
                row["objective_maxdd"] = 999.0
            row.update(trial.user_attrs)
            rows.append(row)

    grid_df = pd.DataFrame(rows)

    # Extract Pareto front
    pareto_trials = study.best_trials
    pareto_rows: list[dict] = []
    for trial in pareto_trials:
        row = dict(trial.params)
        row["cagr"] = trial.values[0] if trial.values[0] != float("-inf") else -999.0
        row["max_drawdown"] = trial.values[1] if trial.values[1] != float("inf") else 999.0
        row.update(trial.user_attrs)
        pareto_rows.append(row)

    pareto_df = pd.DataFrame(pareto_rows) if pareto_rows else None

    log.info(
        "pareto_done",
        universe=universe,
        n_total=len(grid_df),
        n_pareto=len(pareto_rows),
    )

    return MaxProfitResult(
        universe=universe,
        grid_results=grid_df,
        default_metrics=default_metrics,
        default_params=default_params,
        pareto_front=pareto_df,
    )


def select_best_from_pareto(result: MaxProfitResult) -> StrategyParams | None:
    """Select the trial with the best Calmar ratio from the Pareto front."""
    pf = result.pareto_front
    if pf is None or pf.empty:
        return None

    valid = pf[(pf["cagr"] > 0) & (pf["max_drawdown"] > 0)]
    if valid.empty:
        return None

    valid = valid.copy()
    valid["calmar"] = valid["cagr"] / valid["max_drawdown"]
    best = valid.loc[valid["calmar"].idxmax()]

    kwargs = {}
    for key in _MP_PARAM_KEYS:
        if key in best.index:
            kwargs[key] = best[key]
    return StrategyParams(**kwargs)


def format_pareto_report(result: MaxProfitResult, top_n: int = 20) -> str:
    """Format a human-readable Pareto front report."""
    lines: list[str] = []
    lines.append("=" * 90)
    lines.append(f"PARETO FRONT SEARCH — {result.universe.upper()} Universe")
    lines.append("=" * 90)

    # Default baseline
    dm = result.default_metrics
    lines.append("")
    lines.append(f"Default CAGR: {dm['cagr']:.2%}  Max DD: {dm['max_drawdown']:.2%}  "
                 f"Sharpe: {dm['sharpe']:.2f}")

    # Summary
    grid_df = result.grid_results
    valid = grid_df[grid_df.get("objective_cagr", grid_df.get("cagr", pd.Series(dtype=float))) > -999.0]
    lines.append(f"Trials: {len(grid_df)} evaluated, {len(valid)} valid")

    pf = result.pareto_front
    if pf is not None and not pf.empty:
        lines.append(f"Pareto front: {len(pf)} non-dominated solutions")

        lines.append("")
        lines.append("-" * 90)
        lines.append(f"Pareto Front (top {min(top_n, len(pf))} by Calmar ratio):")
        lines.append("-" * 90)

        pf_display = pf[(pf["cagr"] > 0) & (pf["max_drawdown"] > 0)].copy()
        if not pf_display.empty:
            pf_display["calmar"] = pf_display["cagr"] / pf_display["max_drawdown"]
            pf_display = pf_display.nlargest(top_n, "calmar")

            header = (
                f"  {'#':>3} {'kama':>5} {'lbk':>5} {'top_n':>5} {'buf':>7} "
                f"{'CAGR':>8} {'MaxDD':>8} {'Calmar':>8} {'Sharpe':>7}"
            )
            lines.append(header)
            lines.append("  " + "-" * 70)
            for rank, (_, row) in enumerate(pf_display.iterrows(), 1):
                lines.append(
                    f"  {rank:>3} {int(row['kama_period']):>5} "
                    f"{int(row['lookback_period']):>5} "
                    f"{int(row['top_n']):>5} {row['kama_buffer']:>7.4f} "
                    f"{row['cagr']:>8.2%} {row['max_drawdown']:>8.2%} "
                    f"{row['calmar']:>8.2f} {row.get('sharpe', 0):>7.2f}"
                )

    lines.append("")
    lines.append("=" * 90)
    return "\n".join(lines)
```
## File: portfolio_sim/cli_utils.py
```python
"""Shared CLI utilities for entry-point scripts.

Consolidates logging setup, output directory creation, and ticker
filtering that was previously duplicated across run_*.py files.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import structlog


def setup_logging(level: int = logging.INFO) -> None:
    """Configure structlog with console rendering."""
    logging.basicConfig(level=level, format="%(message)s")
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.stdlib.LoggerFactory(),
    )


def create_output_dir(prefix: str) -> Path:
    """Create a timestamped output directory under ``output/``."""
    dt = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output") / f"{prefix}_{dt}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def filter_valid_tickers(
    close_prices: pd.DataFrame,
    min_days: int,
) -> list[str]:
    """Return tickers that have at least *min_days* non-NaN price rows."""
    return [
        t for t in close_prices.columns
        if len(close_prices[t].dropna()) >= min_days
    ]
```
## File: portfolio_sim/params.py
```python
"""Strategy parameter container for walk-forward optimization."""

from dataclasses import dataclass

from src.portfolio_sim.config import (
    CORRELATION_LOOKBACK,
    CORRELATION_THRESHOLD,
    KAMA_BUFFER,
    KAMA_PERIOD,
    LOOKBACK_PERIOD,
    TOP_N,
    VOLATILITY_LOOKBACK,
)


@dataclass(frozen=True)
class StrategyParams:
    """Immutable parameter set for a single backtest run.

    frozen=True makes it hashable, usable as dict keys and in sets.
    Default values match the current config.py constants.
    """

    kama_period: int = KAMA_PERIOD
    lookback_period: int = LOOKBACK_PERIOD
    top_n: int = TOP_N
    kama_buffer: float = KAMA_BUFFER
    use_risk_adjusted: bool = False

    # Market regime filter (SPY-based global kill switch)
    enable_regime_filter: bool = True

    # Correlation filter (greedy diversification)
    enable_correlation_filter: bool = False
    correlation_threshold: float = CORRELATION_THRESHOLD
    correlation_lookback: int = CORRELATION_LOOKBACK

    # Position sizing: "equal_weight" or "risk_parity"
    sizing_mode: str = "equal_weight"
    volatility_lookback: int = VOLATILITY_LOOKBACK

    # Max weight per position (1.0 = no cap)
    max_weight: float = 1.0

    @property
    def warmup(self) -> int:
        """Minimum bars needed before trading can start."""
        return max(
            self.lookback_period,
            self.kama_period,
            self.correlation_lookback,
            self.volatility_lookback,
        ) + 10
```
## File: portfolio_sim/config.py
```python
"""Fixed configuration for simplified KAMA momentum strategy.

All parameters are fixed — no optimization, no tuning.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Trading costs
# ---------------------------------------------------------------------------
INITIAL_CAPITAL: float = 10_000
COMMISSION_RATE: float = 0.0002  # 2 bps (0.02%), Interactive Brokers-like
SLIPPAGE_RATE: float = 0.0005  # 5 bps (0.05%)
RISK_FREE_RATE: float = 0.04

# ---------------------------------------------------------------------------
# Strategy parameters — concentrated momentum (4x S&P 500 target)
# ---------------------------------------------------------------------------
KAMA_PERIOD: int = 20  # Fast adaptive MA (~4 trading weeks)
LOOKBACK_PERIOD: int = 60  # ~3-month momentum window
TOP_N: int = 20  # 20 positions, ~5% each
KAMA_BUFFER: float = 0.008  # 0.8% hysteresis buffer

# ---------------------------------------------------------------------------
# Tickers
# ---------------------------------------------------------------------------
SPY_TICKER: str = "SPY"

# ---------------------------------------------------------------------------
# Cross-asset ETF universe (all-weather tactical allocation)
# ---------------------------------------------------------------------------
ETF_UNIVERSE: list[str] = [
    # US Equities
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "V", "JPM",
    "LLY", "UNH", "MA", "HD", "PG", "XOM", "BAC", "COST", "NFLX", "DIS",
    "KO", "PEP", "NKE", "MCD", "TMO", "ABT", "CRM", "WMT", "WFC", "CMCSA",
    "MRK", "PFE", "CVS", "ALL", "HPQ", "PYPL", "CHTR", "ZS", "HUBS", "PLTR",
    "AMD", "MU", "WDC", "STX", "LRCX", "AMAT", "KLAC", "ADBE", "SHOP", "IREN",

    # International Equities (ADRs)
    "SHEL", "SAP", "ASML", "TSM", "HDB", "SNY", "RYCEY", "AIQUY", "SMFG", "ENB",
    "IBDRY", "SMEGF", "SPOT", "ACN", "ESLOY", "PROSY", "BACHY",

    # Emerging Market Equities (ADRs)
    "BABA", "PDD", "JD", "BIDU", "NTES", "VALE", "MELI", "IBN", "DLO", "GRAB",
    "ARCO", "ABEV", "YUMC", "CPNG", "SEA", "TME", "LI", "NIO", "BEKE", "EDU",

    # US Sector & Thematic ETFs
    "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLU", "XLY", "XLP",
    "XLC", "XLB", "XLRE", "ITA", "VFH", "XRT", "VGT", "XLA", "XAR", "IBB", "SMH", "KBE",

    # International & Regional ETFs
    "VEA", "VXUS", "IEFA", "SCHF", "VPL", "BBEU", "VGK", "EWG", "EWJ", "EWQ",
    "EEM", "VWO", "IEMG", "ILF", "AIA", "EEMA", "FLEU", "SCHY", "VIGI", "AVEM",

    # Bonds & Fixed Income
    "TLT", "IEF", "SHY", "SGOV", "SHV", "LQD", "VCIT", "VTC", "HYG", "JNK",
    "USHY", "FBND", "FIGB", "JCPB", "EMB", "PCY", "BWX", "BSJO", "TLH", "IGIB",

    # Commodities & Metals
    "GLD", "SLV", "IAU", "SGOL", "SIVR", "PPLT", "USO", "BNO", "DBC", "GSG",
    "PDBC", "DJP", "DBA", "UNG", "CORN", "WEAT", "SOYB", "JO", "JJI", "GCC",

    # Real Estate & REITs
    "VNQ", "SCHH", "RWR", "USRT", "REZ", "FRI", "AREA", "RWO", "VNQI", "BBRE",
    "REM", "SRVR", "MORT", "KBWY", "REET", "DFAR", "BCREX", "CSCIX", "PSTL", "ILPT",

    # Crypto (Yahoo Finance format)
    "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD", "DOGE-USD",
    "TRX-USD", "DOT-USD", "LINK-USD", "AVAX-USD", "MATIC-USD", "LTC-USD", "BCH-USD",
    "SHIB-USD", "SUI-USD", "NEAR-USD", "APT-USD", "HBAR-USD", "ONDO-USD",
]

ASSET_CLASS_MAP: dict[str, str] = {
    # US Equities
    "AAPL": "US Equity", "MSFT": "US Equity", "GOOGL": "US Equity", "AMZN": "US Equity", "NVDA": "US Equity",
    "META": "US Equity", "TSLA": "US Equity", "BRK-B": "US Equity", "V": "US Equity", "JPM": "US Equity",
    "LLY": "US Equity", "UNH": "US Equity", "MA": "US Equity", "HD": "US Equity", "PG": "US Equity",
    "XOM": "US Equity", "BAC": "US Equity", "COST": "US Equity", "NFLX": "US Equity", "DIS": "US Equity",
    "KO": "US Equity", "PEP": "US Equity", "NKE": "US Equity", "MCD": "US Equity", "TMO": "US Equity",
    "ABT": "US Equity", "CRM": "US Equity", "WMT": "US Equity", "WFC": "US Equity", "CMCSA": "US Equity",
    "MRK": "US Equity", "PFE": "US Equity", "CVS": "US Equity", "ALL": "US Equity", "HPQ": "US Equity",
    "PYPL": "US Equity", "CHTR": "US Equity", "ZS": "US Equity", "HUBS": "US Equity", "PLTR": "US Equity",
    "AMD": "US Equity", "MU": "US Equity", "WDC": "US Equity", "STX": "US Equity", "LRCX": "US Equity",
    "AMAT": "US Equity", "KLAC": "US Equity", "ADBE": "US Equity", "SHOP": "US Equity", "IREN": "US Equity",

    # International Equities
    "SHEL": "Intl Equity", "SAP": "Intl Equity", "ASML": "Intl Equity", "TSM": "Intl Equity", "HDB": "Intl Equity",
    "SNY": "Intl Equity", "RYCEY": "Intl Equity", "AIQUY": "Intl Equity", "SMFG": "Intl Equity", "ENB": "Intl Equity",
    "IBDRY": "Intl Equity", "SMEGF": "Intl Equity", "SPOT": "Intl Equity", "ACN": "Intl Equity", "ESLOY": "Intl Equity",
    "PROSY": "Intl Equity", "BACHY": "Intl Equity",

    # Emerging Market Equities
    "BABA": "EM Equity", "PDD": "EM Equity", "JD": "EM Equity", "BIDU": "EM Equity", "NTES": "EM Equity",
    "VALE": "EM Equity", "MELI": "EM Equity", "IBN": "EM Equity", "DLO": "EM Equity", "GRAB": "EM Equity",
    "ARCO": "EM Equity", "ABEV": "EM Equity", "YUMC": "EM Equity", "CPNG": "EM Equity", "SEA": "EM Equity",
    "TME": "EM Equity", "LI": "EM Equity", "NIO": "EM Equity", "BEKE": "EM Equity", "EDU": "EM Equity",

    # US Sector ETFs
    "SPY": "US Equity", "QQQ": "US Equity", "XLK": "US Sector ETF", "XLF": "US Sector ETF", "XLE": "US Sector ETF", "XLV": "US Sector ETF", "XLI": "US Sector ETF",
    "XLU": "US Sector ETF", "XLY": "US Sector ETF", "XLP": "US Sector ETF", "XLC": "US Sector ETF", "XLB": "US Sector ETF",
    "XLRE": "US Sector ETF", "ITA": "US Sector ETF", "VFH": "US Sector ETF", "XRT": "US Sector ETF", "VGT": "US Sector ETF",
    "XLA": "US Sector ETF", "XAR": "US Sector ETF", "IBB": "US Sector ETF", "SMH": "US Sector ETF", "KBE": "US Sector ETF",

    # International ETFs
    "VEA": "Intl ETF", "VXUS": "Intl ETF", "IEFA": "Intl ETF", "SCHF": "Intl ETF", "VPL": "Intl ETF",
    "BBEU": "Intl ETF", "VGK": "Intl ETF", "EWG": "Intl ETF", "EWJ": "Intl ETF", "EWQ": "Intl ETF",
    "EEM": "Intl ETF", "VWO": "Intl ETF", "IEMG": "Intl ETF", "ILF": "Intl ETF", "AIA": "Intl ETF",
    "EEMA": "Intl ETF", "FLEU": "Intl ETF", "SCHY": "Intl ETF", "VIGI": "Intl ETF", "AVEM": "Intl ETF",

    # Bonds
    "TLT": "Long Bonds", "IEF": "Mid Bonds", "SHY": "Short Bonds", "SGOV": "Short Bonds", "SHV": "Short Bonds",
    "LQD": "Corporate Bonds", "VCIT": "Corporate Bonds", "VTC": "Corporate Bonds", "HYG": "Corporate Bonds", "JNK": "Corporate Bonds",
    "USHY": "Corporate Bonds", "FBND": "Corporate Bonds", "FIGB": "Corporate Bonds", "JCPB": "Corporate Bonds", "EMB": "Corporate Bonds",
    "PCY": "Corporate Bonds", "BWX": "Corporate Bonds", "BSJO": "Corporate Bonds", "TLH": "Long Bonds", "IGIB": "Corporate Bonds",

    # Commodities
    "GLD": "Commodities", "SLV": "Commodities", "IAU": "Commodities", "SGOL": "Commodities", "SIVR": "Commodities",
    "PPLT": "Commodities", "USO": "Commodities", "BNO": "Commodities", "DBC": "Commodities", "GSG": "Commodities",
    "PDBC": "Commodities", "DJP": "Commodities", "DBA": "Commodities", "UNG": "Commodities", "CORN": "Commodities",
    "WEAT": "Commodities", "SOYB": "Commodities", "JO": "Commodities", "JJI": "Commodities", "GCC": "Commodities",

    # Real Estate
    "VNQ": "Real Estate", "SCHH": "Real Estate", "RWR": "Real Estate", "USRT": "Real Estate", "REZ": "Real Estate",
    "FRI": "Real Estate", "AREA": "Real Estate", "RWO": "Real Estate", "VNQI": "Real Estate", "BBRE": "Real Estate",
    "REM": "Real Estate", "SRVR": "Real Estate", "MORT": "Real Estate", "KBWY": "Real Estate", "REET": "Real Estate",
    "DFAR": "Real Estate", "BCREX": "Real Estate", "CSCIX": "Real Estate", "PSTL": "Real Estate", "ILPT": "Real Estate",

    # Crypto
    "BTC-USD": "Crypto", "ETH-USD": "Crypto", "SOL-USD": "Crypto", "BNB-USD": "Crypto", "XRP-USD": "Crypto",
    "ADA-USD": "Crypto", "DOGE-USD": "Crypto", "TRX-USD": "Crypto", "DOT-USD": "Crypto", "LINK-USD": "Crypto",
    "AVAX-USD": "Crypto", "MATIC-USD": "Crypto", "LTC-USD": "Crypto", "BCH-USD": "Crypto", "SHIB-USD": "Crypto",
    "SUI-USD": "Crypto", "NEAR-USD": "Crypto", "APT-USD": "Crypto", "HBAR-USD": "Crypto", "ONDO-USD": "Crypto",
}

# ---------------------------------------------------------------------------
# Correlation filter (greedy diversification)
# ---------------------------------------------------------------------------
CORRELATION_THRESHOLD: float = 0.9
CORRELATION_LOOKBACK: int = 60  # trading days

# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------
VOLATILITY_LOOKBACK: int = 20  # trading days for inverse-vol weighting

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_DIR: Path = Path("output")
CACHE_DIR: Path = DEFAULT_OUTPUT_DIR / "cache"
```
## File: portfolio_sim/models.py
```python
"""Data models for simulation results."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.portfolio_sim.params import StrategyParams


@dataclass
class SimulationResult:
    """Complete output from a portfolio simulation run."""

    equity: pd.Series
    spy_equity: pd.Series
    holdings_history: pd.DataFrame  # DatetimeIndex x tickers, values = share counts
    cash_history: pd.Series  # daily cash balance
    regime_history: pd.Series | None  # daily bool: True = bull, False = bear (None when regime filter disabled)
    trade_log: list[dict] = field(default_factory=list)
    # Each entry: {"date": date, "ticker": str, "action": "buy"|"sell"|"liquidate",
    #              "shares": float, "price": float}


@dataclass
class WFOStep:
    """One step of walk-forward optimization."""

    step_index: int
    is_start: pd.Timestamp
    is_end: pd.Timestamp
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp
    optimized_params: StrategyParams
    is_metrics: dict
    oos_metrics: dict
    oos_equity: pd.Series  # raw OOS equity curve (starts at initial_capital)
    oos_spy_equity: pd.Series  # SPY equity over same OOS period


@dataclass
class WFOResult:
    """Complete walk-forward optimization result."""

    steps: list[WFOStep]
    stitched_equity: pd.Series  # concatenated OOS equity curves
    stitched_spy_equity: pd.Series  # concatenated SPY equity for same OOS periods
    oos_metrics: dict  # metrics computed on stitched OOS equity
    final_params: StrategyParams  # params from the last IS window (for live use)
```
## File: portfolio_sim/indicators.py
```python
"""Kaufman's Adaptive Moving Average (KAMA) — Numba JIT accelerated.

Standalone implementation (no external config dependencies).
"""

import numpy as np
import pandas as pd
from numba import njit


@njit(cache=True)
def _kama_recurrent_loop(
    data_array: np.ndarray,
    smoothing_array: np.ndarray,
    kama: np.ndarray,
    period: int,
) -> np.ndarray:
    """Numba JIT recurrent KAMA update. cache=True for zero warm-up on restart."""
    n = len(data_array)
    for i in range(period + 1, n):
        smoothing = smoothing_array[i - period]
        kama[i] = kama[i - 1] + smoothing * (data_array[i] - kama[i - 1])
    return kama


def compute_kama(
    data: np.ndarray | pd.Series,
    period: int = 20,
    fast_constant: int = 2,
    slow_constant: int = 30,
) -> np.ndarray:
    """Compute Kaufman Adaptive Moving Average.

    Returns np.ndarray of KAMA values (NaN-padded at the start).
    """
    period = max(1, int(period))
    fast_constant = max(1, int(fast_constant))
    slow_constant = max(1, int(slow_constant))

    data_array = np.asarray(data, dtype=float)
    n = len(data_array)

    if n <= period:
        return np.full(n, np.nan, dtype=float)

    kama = np.full(n, np.nan, dtype=float)

    fast_const = 2.0 / (fast_constant + 1)
    slow_const = 2.0 / (slow_constant + 1)

    # Vectorized price change: abs(data[i] - data[i - period])
    price_change = np.abs(data_array[period:] - data_array[:-period])

    # Vectorized volatility: rolling sum of abs(diff(data)) over period
    abs_diff = np.abs(np.diff(data_array))
    abs_diff_cumsum = np.concatenate(([0], np.cumsum(abs_diff)))
    volatility = abs_diff_cumsum[period:] - abs_diff_cumsum[:-period]

    # Efficiency Ratio
    with np.errstate(divide="ignore", invalid="ignore"):
        er = np.where(volatility > 0, price_change / volatility, 0.0)
    er = np.clip(er, 0, 1)

    # Smoothing constant
    sc = (er * (fast_const - slow_const) + slow_const) ** 2
    smoothing_array = np.clip(sc, 0, 1)

    # Initial KAMA value
    kama[period] = data_array[period]

    # Recurrent loop via Numba JIT
    kama = _kama_recurrent_loop(data_array, smoothing_array, kama, period)

    return kama


def compute_kama_series(prices: pd.Series, period: int = 20) -> pd.Series:
    """Convenience wrapper returning pd.Series with the same index as input."""
    kama_arr = compute_kama(prices.values, period=period)
    return pd.Series(kama_arr, index=prices.index)
```
## File: portfolio_sim/engine.py
```python
"""Bar-by-bar simulation engine — Long/Cash only.

Signals computed on Close(T), execution on Open(T+1).

Key design decisions that prevent capital destruction:
  - Market Breathing with hysteresis: SPY must cross KAMA by ±KAMA_BUFFER to
    flip regime, preventing sell-all/buy-all churn in sideways markets.
    (Controlled by enable_regime_filter; disabled for cross-asset ETF mode.)
  - Lazy hold: positions are sold ONLY when their own KAMA stop-loss triggers,
    never because they dropped in the momentum ranking.
  - Position sizing: strict 1/TOP_N (equal weight) OR inverse-volatility
    (risk parity), controlled by sizing_mode parameter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.portfolio_sim.alpha import get_buy_candidates
from src.portfolio_sim.config import (
    COMMISSION_RATE,
    SLIPPAGE_RATE,
    SPY_TICKER,
)
from src.portfolio_sim.indicators import compute_kama_series
from src.portfolio_sim.models import SimulationResult
from src.portfolio_sim.params import StrategyParams

COST_RATE = COMMISSION_RATE + SLIPPAGE_RATE


def run_simulation(
    close_prices: pd.DataFrame,
    open_prices: pd.DataFrame,
    tickers: list[str],
    initial_capital: float,
    params: StrategyParams | None = None,
    kama_cache: dict[str, pd.Series] | None = None,
    show_progress: bool = False,
) -> SimulationResult:
    """Run a full bar-by-bar portfolio simulation.

    Args:
        close_prices: Full history of Close prices (DatetimeIndex rows,
                      ticker columns). Must include SPY.
        open_prices: Full history of Open prices (same shape/index).
        tickers: list of tradable ticker symbols.
        initial_capital: starting cash.
        params: strategy parameters. Falls back to config defaults when *None*.
        kama_cache: pre-computed {ticker: kama_series}. Computed internally
                    when *None*.

    Returns:
        SimulationResult with equity curves, holdings history, regime, and trades.
    """
    p = params or StrategyParams()
    # ------------------------------------------------------------------
    # 0. Pre-compute KAMA for all tickers + SPY (skip if caller provided)
    # ------------------------------------------------------------------
    if kama_cache is None:
        kama_cache = {}
        all_tickers = list(set(tickers) | {SPY_TICKER})
        ticker_iter = tqdm(all_tickers, desc="Computing KAMA", unit="ticker") if show_progress else all_tickers
        for t in ticker_iter:
            if t in close_prices.columns:
                kama_cache[t] = compute_kama_series(
                    close_prices[t].dropna(), period=p.kama_period
                )

    # ------------------------------------------------------------------
    # 1. Determine simulation start (need warm-up for KAMA + lookback)
    # ------------------------------------------------------------------
    warmup = p.warmup
    if len(close_prices) <= warmup:
        raise ValueError(f"Need at least {warmup} rows, got {len(close_prices)}")

    sim_dates = close_prices.index[warmup:]

    spy_prices = close_prices[SPY_TICKER].loc[sim_dates]
    spy_equity = initial_capital * (spy_prices / spy_prices.iloc[0])

    # ------------------------------------------------------------------
    # 2. State
    # ------------------------------------------------------------------
    cash = initial_capital
    shares: dict[str, float] = {}
    equity_values: list[float] = []

    pending_trades: dict[str, float] | None = None
    pending_weights: dict[str, float] | None = None  # risk parity weights for pending buys
    is_bull = True  # regime memory — persists across days (hysteresis)

    # Tracking for dashboard
    cash_values: list[float] = []
    regime_values: list[bool] = [] if p.enable_regime_filter else None
    holdings_rows: list[dict[str, float]] = []
    trade_log: list[dict] = []

    # ------------------------------------------------------------------
    # 3. Day-by-day loop
    # ------------------------------------------------------------------
    date_iter = tqdm(sim_dates, desc="Simulating", unit="day") if show_progress else sim_dates
    for i, date in enumerate(date_iter):
        daily_close = close_prices.loc[date]
        daily_open = open_prices.loc[date]

        # --- Execute pending trades on Open ---
        if pending_trades is not None:
            equity_at_open = cash + sum(
                shares[t] * daily_open.get(t, 0.0) for t in shares
            )
            if equity_at_open <= 0:
                equity_at_open = max(cash, 1.0)

            shares_before = set(shares.keys())
            cash = _execute_trades(
                shares, pending_trades, equity_at_open, daily_open,
                p.top_n, weights=pending_weights, max_weight=p.max_weight,
            )

            # Record trades
            for t in shares_before - set(shares.keys()):
                trade_log.append({
                    "date": date, "ticker": t, "action": "sell",
                    "shares": 0.0, "price": daily_open.get(t, 0.0),
                })
            for t in set(shares.keys()) - shares_before:
                trade_log.append({
                    "date": date, "ticker": t, "action": "buy",
                    "shares": shares[t], "price": daily_open.get(t, 0.0),
                })
            pending_trades = None
            pending_weights = None

        # --- Mark-to-market on Close ---
        equity_value = cash + sum(
            shares[t] * daily_close.get(t, 0.0) for t in shares
        )
        equity_values.append(equity_value)

        # Record daily snapshot
        cash_values.append(cash)
        if regime_values is not None:
            regime_values.append(is_bull)
        holdings_rows.append({t: shares.get(t, 0.0) for t in tickers})

        if equity_value <= 0:
            remaining = len(sim_dates) - i - 1
            equity_values.extend([0.0] * remaining)
            cash_values.extend([0.0] * remaining)
            if regime_values is not None:
                regime_values.extend([False] * remaining)
            empty_row = {t: 0.0 for t in tickers}
            holdings_rows.extend([empty_row] * remaining)
            break

        # --- Compute signals on Close(T) for Open(T+1) ---

        # Market Breathing with hysteresis buffer (optional regime filter).
        if p.enable_regime_filter:
            spy_close = daily_close.get(SPY_TICKER, np.nan)
            spy_kama_s = kama_cache.get(SPY_TICKER, pd.Series(dtype=float))
            spy_kama = spy_kama_s.get(date, np.nan) if date in spy_kama_s.index else np.nan

            if not np.isnan(spy_close) and not np.isnan(spy_kama):
                if is_bull and spy_close < spy_kama * (1 - p.kama_buffer):
                    is_bull = False
                elif not is_bull and spy_close > spy_kama * (1 + p.kama_buffer):
                    is_bull = True

            if not is_bull:
                if shares:
                    pending_trades = {}
                continue

        # FIX #1: Sell ONLY on individual KAMA stop-loss.
        # A stock dropping from rank 20 to rank 50 is irrelevant
        # as long as its own trend (Close > KAMA) holds.
        sells: dict[str, float] = {}
        for t in list(shares.keys()):
            t_kama_s = kama_cache.get(t, pd.Series(dtype=float))
            t_kama = t_kama_s.get(date, np.nan) if date in t_kama_s.index else np.nan
            if not np.isnan(t_kama):
                if daily_close.get(t, 0.0) < t_kama * (1 - p.kama_buffer):
                    sells[t] = 0.0

        # Get fresh candidates from alpha
        idx_in_full = close_prices.index.get_loc(date)
        start = max(0, idx_in_full - p.lookback_period + 1)
        past_prices = close_prices.iloc[start : idx_in_full + 1][tickers]

        kama_current: dict[str, float] = {}
        for t in tickers:
            if t in kama_cache:
                val = (
                    kama_cache[t].get(date, np.nan)
                    if date in kama_cache[t].index
                    else np.nan
                )
                if not np.isnan(val):
                    kama_current[t] = val

        candidates = get_buy_candidates(
            past_prices, tickers, kama_current,
            kama_buffer=p.kama_buffer, top_n=p.top_n,
            use_risk_adjusted=p.use_risk_adjusted,
            correlation_threshold=p.correlation_threshold,
            correlation_lookback=p.correlation_lookback,
            enable_correlation_filter=p.enable_correlation_filter,
        )

        # Build trade instructions
        new_trades: dict[str, float] = {}

        for t in sells:
            new_trades[t] = 0.0

        # FIX #1 (continued): fill only empty slots with new candidates.
        # Held positions that lost rank but kept their trend stay untouched.
        open_slots = p.top_n - (len(shares) - len(sells))

        if open_slots > 0:
            for t in candidates:
                if t not in shares and t not in sells:
                    new_trades[t] = 1.0
                    open_slots -= 1
                    if open_slots <= 0:
                        break

        if new_trades:
            pending_trades = new_trades

            # Compute risk parity weights for new buys
            if p.sizing_mode == "risk_parity":
                buys_list = [t for t, v in new_trades.items() if v == 1.0]
                if buys_list:
                    vol_start = max(0, idx_in_full - p.volatility_lookback)
                    vol_prices = close_prices.iloc[vol_start : idx_in_full + 1]
                    pending_weights = _compute_inverse_vol_weights(
                        buys_list, vol_prices, p.volatility_lookback,
                        max_weight=p.max_weight,
                    )

    n = len(equity_values)
    idx = sim_dates[:n]
    equity_series = pd.Series(equity_values, index=idx)
    spy_series = spy_equity.iloc[:n]

    holdings_df = pd.DataFrame(holdings_rows, index=idx)
    cash_series = pd.Series(cash_values, index=idx)
    regime_series = (
        pd.Series(regime_values, index=idx, dtype=bool)
        if regime_values is not None
        else None
    )

    return SimulationResult(
        equity=equity_series,
        spy_equity=spy_series,
        holdings_history=holdings_df,
        cash_history=cash_series,
        regime_history=regime_series,
        trade_log=trade_log,
    )


def _cap_and_redistribute(
    weights: dict[str, float], max_weight: float,
) -> dict[str, float]:
    """Cap each weight at *max_weight* and redistribute excess proportionally.

    Iterative: after redistribution some weights may exceed the cap again,
    so we repeat until convergence (at most N rounds).
    """
    if max_weight >= 1.0:
        return weights

    result = dict(weights)
    for _ in range(len(result)):
        excess = 0.0
        uncapped_total = 0.0
        for t, w in result.items():
            if w > max_weight:
                excess += w - max_weight
                result[t] = max_weight
            elif w < max_weight:
                uncapped_total += w
            # w == max_weight: already at cap, skip

        if excess < 1e-10:
            break

        if uncapped_total > 0:
            for t in result:
                if result[t] < max_weight:
                    result[t] += excess * (result[t] / uncapped_total)
        else:
            break

    return result


def _compute_inverse_vol_weights(
    tickers_to_buy: list[str],
    close_prices_window: pd.DataFrame,
    volatility_lookback: int = 20,
    max_weight: float = 1.0,
) -> dict[str, float]:
    """Compute inverse-volatility weights for a set of tickers.

    Each ticker's weight is proportional to 1/volatility, so low-vol assets
    (e.g. bonds) get larger allocations and high-vol assets (e.g. EM equities)
    get smaller allocations. This ensures each position contributes roughly
    equal dollar risk to the portfolio.

    Falls back to equal weight if volatility cannot be computed.
    """
    if not tickers_to_buy:
        return {}

    inv_vols: dict[str, float] = {}
    for t in tickers_to_buy:
        if t not in close_prices_window.columns:
            continue
        recent = close_prices_window[t].iloc[-volatility_lookback:]
        returns = recent.pct_change().dropna()
        if len(returns) < 5:
            continue
        vol = returns.std()
        if vol > 1e-8:
            inv_vols[t] = 1.0 / vol
        else:
            inv_vols[t] = 1.0  # near-zero vol: treat as equal weight

    if not inv_vols:
        return {t: 1.0 / len(tickers_to_buy) for t in tickers_to_buy}

    total = sum(inv_vols.values())
    weights = {t: v / total for t, v in inv_vols.items()}
    return _cap_and_redistribute(weights, max_weight)


def _execute_trades(
    shares: dict[str, float],
    trades: dict[str, float],
    equity_at_open: float,
    open_prices: pd.Series,
    top_n: int = StrategyParams().top_n,
    weights: dict[str, float] | None = None,
    max_weight: float = 1.0,
) -> float:
    """Execute trades. Mutates ``shares`` in place. Returns remaining cash.

    Convention:
      - trades == {} (empty dict): sell everything (bear regime).
      - trades[t] == 0.0: sell position t.
      - trades[t] == 1.0: buy position t.

    When *weights* is provided (risk parity mode), each buy's allocation is
    equity_at_open * weights[t]. Otherwise strict 1/top_n equal weight.
    *max_weight* caps any single position at equity_at_open * max_weight.
    """
    total_cost = 0.0

    # Bear regime: liquidate everything
    if not trades:
        for t in list(shares.keys()):
            price = open_prices.get(t, 0.0)
            if price > 0 and shares[t] > 0:
                trade_value = shares[t] * price
                total_cost += trade_value * COST_RATE
            del shares[t]
        return equity_at_open - total_cost

    # Execute individual stop-loss sells
    for t, action in list(trades.items()):
        if action == 0.0 and t in shares:
            price = open_prices.get(t, 0.0)
            if price > 0:
                trade_value = shares[t] * price
                total_cost += trade_value * COST_RATE
            del shares[t]

    held_value = sum(shares[t] * open_prices.get(t, 0.0) for t in shares)
    available = equity_at_open - held_value - total_cost

    # Position sizing: risk parity (inverse vol) or equal weight (1/TOP_N).
    buys = [t for t, action in trades.items() if action == 1.0 and t not in shares]
    if buys and available > 0:
        for t in buys:
            price = open_prices.get(t, 0.0)
            if weights is not None and t in weights:
                max_allocation = equity_at_open * weights[t]
            else:
                max_allocation = min(
                    equity_at_open / top_n,
                    equity_at_open * max_weight,
                )
            allocation = min(max_allocation, available)
            if price > 0 and allocation > 0:
                net_investment = allocation / (1 + COST_RATE)
                cost = net_investment * COST_RATE
                shares[t] = net_investment / price
                total_cost += cost
                available -= allocation

    held_value = sum(shares[t] * open_prices.get(t, 0.0) for t in shares)
    return equity_at_open - held_value - total_cost
```
## File: portfolio_sim/cli.py
```python
"""CLI dispatcher with subcommands for portfolio_sim."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="portfolio",
        description="KAMA Momentum Strategy toolkit",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    from src.portfolio_sim.commands import (
        max_profit,
        optimize,
        walk_forward,
    )

    commands: dict[str, object] = {}
    for mod in [optimize, max_profit, walk_forward]:
        mod.register(subparsers)
        commands[mod.COMMAND_NAME] = mod.run

    args = parser.parse_args(argv)
    commands[args.command](args)
```
## File: portfolio_sim/alpha.py
```python
"""KAMA momentum alpha — Long/Cash only.

Selects buy candidates from the universe based on:
  1. KAMA trend filter: keep stocks where Close > KAMA * (1 + KAMA_BUFFER).
  2. Risk-adjusted momentum ranking: sort by return/volatility (Sharpe-like
     momentum), take top N.  This prefers strong AND smooth uptrends.
  3. (Optional) Greedy correlation filter: skip candidates that are too
     correlated with assets already selected, ensuring diversification.

Does NOT dictate weights — sizing is handled by the engine.
"""

import numpy as np
import pandas as pd

from src.portfolio_sim.config import (
    CORRELATION_LOOKBACK,
    CORRELATION_THRESHOLD,
    KAMA_BUFFER,
    TOP_N,
)


def _greedy_correlation_filter(
    ranked_tickers: list[str],
    prices_window: pd.DataFrame,
    top_n: int,
    correlation_threshold: float,
    correlation_lookback: int,
) -> list[str]:
    """Greedy diversification: select tickers one-by-one, skipping those
    too correlated with already-selected assets.

    Args:
        ranked_tickers: tickers sorted by momentum score (descending).
        prices_window: price DataFrame for computing return correlations.
        top_n: maximum basket size.
        correlation_threshold: max allowed absolute pairwise correlation.
        correlation_lookback: days of returns to use for correlation.

    Returns:
        Filtered list of up to *top_n* diversified tickers.
    """
    if not ranked_tickers:
        return []

    # Use only columns that exist in the price window
    available = [t for t in ranked_tickers if t in prices_window.columns]
    if not available:
        return []

    recent_prices = prices_window[available].iloc[-correlation_lookback:]
    returns = recent_prices.pct_change().dropna()

    if len(returns) < 10:
        # Not enough data for meaningful correlation — skip filter
        return ranked_tickers[:top_n]

    basket: list[str] = []

    for ticker in ranked_tickers:
        if len(basket) >= top_n:
            break

        if ticker not in returns.columns:
            continue

        if not basket:
            basket.append(ticker)
            continue

        # Check correlation with each basket member
        too_correlated = False
        for held in basket:
            if held not in returns.columns:
                continue
            corr = returns[ticker].corr(returns[held])
            if not np.isnan(corr) and abs(corr) > correlation_threshold:
                too_correlated = True
                break

        if not too_correlated:
            basket.append(ticker)

    return basket


def get_buy_candidates(
    prices_window: pd.DataFrame,
    tickers: list[str],
    kama_values: dict[str, float],
    kama_buffer: float = KAMA_BUFFER,
    top_n: int = TOP_N,
    use_risk_adjusted: bool = True,
    correlation_threshold: float = CORRELATION_THRESHOLD,
    correlation_lookback: int = CORRELATION_LOOKBACK,
    enable_correlation_filter: bool = False,
) -> list[str]:
    """Return an ordered list of top-momentum tickers passing the KAMA filter.

    Args:
        prices_window: Close prices with rows >= LOOKBACK_PERIOD trading days,
                       cols = tickers.
        tickers: ordered list of tradable ticker symbols.
        kama_values: {ticker: current_kama_value} for KAMA filter.
        kama_buffer: hysteresis buffer for KAMA filter (default from config).
        top_n: maximum number of candidates to return (default from config).
        use_risk_adjusted: if True, rank by return/volatility instead of raw
                           return.  Prefers smooth uptrends.
        correlation_threshold: max allowed pairwise correlation for greedy filter.
        correlation_lookback: days of returns for correlation computation.
        enable_correlation_filter: if True, apply greedy correlation diversification
                                   after momentum ranking.

    Returns:
        List of up to *top_n* ticker symbols, ranked by descending score.
        Empty list when no candidates pass both filters.
    """
    candidates = []

    for t in tickers:
        if t not in prices_window.columns:
            continue
        close = prices_window[t].iloc[-1]
        kama = kama_values.get(t, np.nan)
        if np.isnan(close) or np.isnan(kama):
            continue
        if close > kama * (1 + kama_buffer):
            candidates.append(t)

    if not candidates:
        return []

    scores: dict[str, float] = {}
    for t in candidates:
        series = prices_window[t].dropna()
        if len(series) < 5:
            continue
        close_now = series.iloc[-1]
        close_past = series.iloc[0]
        if np.isnan(close_past) or close_past <= 1e-8:
            continue
        raw_return = close_now / close_past - 1.0
        if raw_return <= 0:
            continue

        if use_risk_adjusted:
            daily_returns = series.pct_change().dropna()
            vol = daily_returns.std()
            if vol > 1e-8:
                scores[t] = raw_return / vol
            else:
                scores[t] = raw_return
        else:
            scores[t] = raw_return

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ranked_tickers = [t for t, _ in ranked]

    if enable_correlation_filter:
        return _greedy_correlation_filter(
            ranked_tickers, prices_window, top_n,
            correlation_threshold, correlation_lookback,
        )

    return ranked_tickers[:top_n]
```
## File: portfolio_sim/optimizer.py
```python
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

import numpy as np
import optuna
import pandas as pd
import structlog
from tqdm import tqdm

from src.portfolio_sim.config import INITIAL_CAPITAL, SPY_TICKER
from src.portfolio_sim.indicators import compute_kama_series
from src.portfolio_sim.parallel import (
    _shared,
    evaluate_combo,
    init_eval_worker,
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
# Default search space for sensitivity analysis
# ---------------------------------------------------------------------------
SENSITIVITY_SPACE: dict[str, dict] = {
    "kama_period": {"type": "categorical", "choices": [10, 15, 20, 30, 40]},
    "lookback_period": {"type": "int", "low": 20, "high": 150, "step": 10},
    "kama_buffer": {"type": "float", "low": 0.005, "high": 0.03, "step": 0.001},
    "top_n": {"type": "int", "low": 5, "high": 30, "step": 5},
}

DEFAULT_N_TRIALS: int = 50

PARAM_NAMES: list[str] = ["kama_period", "lookback_period", "kama_buffer", "top_n"]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
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
    if not n_workers or n_workers < 1:
        n_workers = max(1, os.cpu_count() - 1)

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
_SENS_PARAM_KEYS = ["kama_period", "lookback_period", "kama_buffer", "top_n"]
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
    space: dict[str, dict] | None = None,
    n_trials: int = DEFAULT_N_TRIALS,
    n_workers: int | None = None,
    max_dd_limit: float = 0.30,
) -> SensitivityResult:
    """Run parameter sensitivity analysis using Optuna TPE sampler.

    1. Pre-compute KAMA for all possible kama_period values (parallel).
    2. Use Optuna to sample parameter combinations efficiently.
    3. Evaluate each combination in parallel via ProcessPoolExecutor.
    4. Compute 1D marginal profiles (mean objective per parameter value).
    5. Score robustness: flat profile = robust parameter.

    This is NOT optimization — the goal is to verify that performance
    is stable across parameter values, not to find the "best" combo.
    """
    base_params = base_params or StrategyParams()
    space = space or SENSITIVITY_SPACE
    if not n_workers or n_workers < 1:
        n_workers = max(1, os.cpu_count() - 1)

    log.info(
        "sensitivity_start",
        n_trials=n_trials,
        n_workers=n_workers,
    )

    # Pre-compute KAMA for all possible kama_period values
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

    # Use ask/tell API with batch parallelism via ProcessPoolExecutor.
    # This avoids the overhead of Optuna's internal threading (n_jobs)
    # while keeping full process-level parallelism for CPU-bound simulations.
    executor = ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=init_eval_worker,
        initargs=(close_prices, open_prices, tickers, initial_capital, kama_caches),
    )

    pbar = tqdm(total=n_trials, desc="Sensitivity trials", unit="trial")
    trials_done = 0

    while trials_done < n_trials:
        batch_size = min(n_workers, n_trials - trials_done)
        trials = [study.ask() for _ in range(batch_size)]
        params_list = [suggest_params(t, space) for t in trials]

        futures = {
            executor.submit(
                evaluate_combo,
                (p, max_dd_limit, compute_objective, "objective",
                 _SENS_PARAM_KEYS, _SENS_METRIC_KEYS),
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


def find_best_params(result: SensitivityResult) -> StrategyParams | None:
    """Extract the best parameter combo from sensitivity results.

    Returns StrategyParams for the combo with highest objective,
    or None if no valid combo was found.
    """
    valid = result.grid_results[result.grid_results["objective"] > -999.0]
    if valid.empty:
        return None
    best = valid.loc[valid["objective"].idxmax()]
    return StrategyParams(
        kama_period=int(best["kama_period"]),
        lookback_period=int(best["lookback_period"]),
        kama_buffer=float(best["kama_buffer"]),
        top_n=int(best["top_n"]),
    )
```
## File: portfolio_sim/walk_forward.py
```python
"""Walk-Forward Optimization (WFO) for KAMA momentum strategy.

Splits the timeline into expanding in-sample (IS) windows for parameter
optimization and fixed out-of-sample (OOS) windows for validation.
This prevents overfitting by never testing parameters on data they were
trained on.

Anchored WFO: IS always starts at the first available date and grows
by *oos_days* each step.  The final step's optimized parameters are
the recommended "live" parameters.
"""

from __future__ import annotations

import os

import pandas as pd
import structlog

from src.portfolio_sim.config import INITIAL_CAPITAL
from src.portfolio_sim.engine import run_simulation
from src.portfolio_sim.models import WFOResult, WFOStep
from src.portfolio_sim.optimizer import (
    SENSITIVITY_SPACE,
    _get_kama_periods_from_space,
    compute_objective,
    find_best_params,
    precompute_kama_caches,
    run_sensitivity,
)
from src.portfolio_sim.params import StrategyParams
from src.portfolio_sim.reporting import compute_metrics

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Schedule generation
# ---------------------------------------------------------------------------
def generate_wfo_schedule(
    dates: pd.DatetimeIndex,
    min_is_days: int = 756,
    oos_days: int = 252,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Generate (is_start, is_end, oos_start, oos_end) tuples.

    Anchored WFO: IS always starts at ``dates[0]``, IS end advances by
    *oos_days* each step.

    Returns empty list if there is not enough data for even one step.
    """
    total = len(dates)
    schedule: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    is_end_idx = min_is_days - 1

    while is_end_idx + oos_days < total:
        is_start = dates[0]
        is_end = dates[is_end_idx]
        oos_start = dates[is_end_idx + 1]
        oos_end_idx = min(is_end_idx + oos_days, total - 1)
        oos_end = dates[oos_end_idx]

        schedule.append((is_start, is_end, oos_start, oos_end))
        is_end_idx += oos_days

    return schedule


# ---------------------------------------------------------------------------
# Main walk-forward optimization
# ---------------------------------------------------------------------------
def run_walk_forward(
    close_prices: pd.DataFrame,
    open_prices: pd.DataFrame,
    tickers: list[str],
    initial_capital: float = INITIAL_CAPITAL,
    base_params: StrategyParams | None = None,
    space: dict[str, dict] | None = None,
    n_trials_per_step: int = 50,
    n_workers: int | None = None,
    min_is_days: int = 756,
    oos_days: int = 252,
    max_dd_limit: float = 0.30,
) -> WFOResult:
    """Run anchored walk-forward optimization.

    For each step:
      1. Optimize parameters on the IS (in-sample) data slice.
      2. Run simulation with optimized params on the OOS (out-of-sample)
         data slice (with warmup prefix for indicator computation).
      3. Record IS and OOS metrics.

    After all steps the OOS equity curves are stitched together
    to produce the aggregate out-of-sample performance.
    """
    base_params = base_params or StrategyParams()
    space = space or SENSITIVITY_SPACE
    n_workers = n_workers or max(1, os.cpu_count() - 1)

    dates = close_prices.index
    schedule = generate_wfo_schedule(dates, min_is_days, oos_days)

    if not schedule:
        raise ValueError(
            f"Not enough data for WFO: {len(dates)} trading days available, "
            f"need at least {min_is_days + oos_days} "
            f"(min_is_days={min_is_days} + oos_days={oos_days})."
        )

    log.info(
        "wfo_start",
        n_steps=len(schedule),
        min_is_days=min_is_days,
        oos_days=oos_days,
        n_trials_per_step=n_trials_per_step,
    )

    steps: list[WFOStep] = []

    for step_idx, (is_start, is_end, oos_start, oos_end) in enumerate(schedule):
        log.info(
            "wfo_step_start",
            step=step_idx + 1,
            is_range=f"{is_start.date()}..{is_end.date()}",
            oos_range=f"{oos_start.date()}..{oos_end.date()}",
        )

        # --- 1. Slice IS data ---
        close_is = close_prices.loc[is_start:is_end]
        open_is = open_prices.loc[is_start:is_end]
        valid_is = [
            t for t in tickers
            if t in close_is.columns and len(close_is[t].dropna()) >= min_is_days // 2
        ]

        # --- 2. Optimize on IS ---
        sens_result = run_sensitivity(
            close_is, open_is, valid_is,
            initial_capital,
            base_params=base_params,
            space=space,
            n_trials=n_trials_per_step,
            n_workers=n_workers,
            max_dd_limit=max_dd_limit,
        )

        best_params = find_best_params(sens_result)
        if best_params is None:
            log.warning("wfo_step_no_valid_params", step=step_idx + 1)
            best_params = base_params

        # Run IS simulation with best params for IS metrics
        is_sim = run_simulation(
            close_is, open_is, valid_is, initial_capital,
            params=best_params,
        )
        is_metrics = compute_metrics(is_sim.equity)

        # --- 3. Run OOS simulation with warmup prefix ---
        warmup = best_params.warmup
        oos_start_loc = dates.get_loc(oos_start)
        warmup_start_idx = max(0, oos_start_loc - warmup)
        oos_end_loc = dates.get_loc(oos_end)

        close_oos_warm = close_prices.iloc[warmup_start_idx:oos_end_loc + 1]
        open_oos_warm = open_prices.iloc[warmup_start_idx:oos_end_loc + 1]

        valid_oos = [
            t for t in tickers
            if t in close_oos_warm.columns and len(close_oos_warm[t].dropna()) >= warmup
        ]

        oos_sim = run_simulation(
            close_oos_warm, open_oos_warm, valid_oos, initial_capital,
            params=best_params,
        )

        # Trim equity to only the OOS period
        oos_equity = oos_sim.equity.loc[oos_start:oos_end]
        oos_spy_equity = oos_sim.spy_equity.loc[oos_start:oos_end]

        if oos_equity.empty:
            log.warning("wfo_step_empty_oos", step=step_idx + 1)
            continue

        oos_metrics = compute_metrics(oos_equity)

        step = WFOStep(
            step_index=step_idx,
            is_start=is_start,
            is_end=is_end,
            oos_start=oos_start,
            oos_end=oos_end,
            optimized_params=best_params,
            is_metrics=is_metrics,
            oos_metrics=oos_metrics,
            oos_equity=oos_equity,
            oos_spy_equity=oos_spy_equity,
        )
        steps.append(step)

        log.info(
            "wfo_step_done",
            step=step_idx + 1,
            is_cagr=f"{is_metrics.get('cagr', 0):.2%}",
            oos_cagr=f"{oos_metrics.get('cagr', 0):.2%}",
        )

    if not steps:
        raise ValueError("WFO produced no valid steps.")

    # --- 4. Stitch OOS equity curves ---
    stitched_equity = _stitch_equity_curves(
        [s.oos_equity for s in steps], initial_capital,
    )
    stitched_spy = _stitch_equity_curves(
        [s.oos_spy_equity for s in steps], initial_capital,
    )

    oos_metrics = compute_metrics(stitched_equity)
    final_params = steps[-1].optimized_params

    log.info(
        "wfo_done",
        n_steps=len(steps),
        oos_cagr=f"{oos_metrics.get('cagr', 0):.2%}",
        oos_maxdd=f"{oos_metrics.get('max_drawdown', 0):.2%}",
    )

    return WFOResult(
        steps=steps,
        stitched_equity=stitched_equity,
        stitched_spy_equity=stitched_spy,
        oos_metrics=oos_metrics,
        final_params=final_params,
    )


# ---------------------------------------------------------------------------
# Equity curve stitching
# ---------------------------------------------------------------------------
def _stitch_equity_curves(
    curves: list[pd.Series],
    initial_capital: float,
) -> pd.Series:
    """Concatenate OOS equity curves, scaling each segment.

    Each segment is scaled so it starts where the previous segment ended.
    The first segment is scaled to start at *initial_capital*.
    """
    segments: list[pd.Series] = []
    current_value = initial_capital

    for curve in curves:
        if curve.empty:
            continue
        scale = current_value / curve.iloc[0]
        scaled = curve * scale
        segments.append(scaled)
        current_value = scaled.iloc[-1]

    if not segments:
        return pd.Series(dtype=float)

    return pd.concat(segments)


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------
def format_wfo_report(result: WFOResult) -> str:
    """Format a human-readable walk-forward optimization report."""
    lines: list[str] = []
    lines.append("=" * 90)
    lines.append("WALK-FORWARD OPTIMIZATION REPORT")
    lines.append("=" * 90)

    lines.append("")
    lines.append(f"Schedule: Anchored WFO, {len(result.steps)} steps")
    if result.steps:
        s0 = result.steps[0]
        lines.append(f"  Full period: {s0.is_start.date()} .. {result.steps[-1].oos_end.date()}")

    # Per-step table
    lines.append("")
    lines.append("-" * 90)
    header = (
        f"  {'Step':>4}  {'IS Period':<25} {'OOS Period':<25}"
        f"  {'IS CAGR':>8}  {'OOS CAGR':>9}  {'OOS MaxDD':>9}"
    )
    lines.append(header)
    lines.append("-" * 90)

    is_cagrs = []
    oos_cagrs = []
    for step in result.steps:
        is_cagr = step.is_metrics.get("cagr", 0.0)
        oos_cagr = step.oos_metrics.get("cagr", 0.0)
        oos_maxdd = step.oos_metrics.get("max_drawdown", 0.0)
        is_cagrs.append(is_cagr)
        oos_cagrs.append(oos_cagr)

        is_range = f"{step.is_start.date()}..{step.is_end.date()}"
        oos_range = f"{step.oos_start.date()}..{step.oos_end.date()}"

        lines.append(
            f"  {step.step_index + 1:>4}  {is_range:<25} {oos_range:<25}"
            f"  {is_cagr:>8.2%}  {oos_cagr:>9.2%}  {oos_maxdd:>9.2%}"
        )

    # Stitched OOS performance
    lines.append("")
    lines.append("-" * 90)
    lines.append("Stitched Out-of-Sample Performance:")
    lines.append("-" * 90)

    om = result.oos_metrics
    lines.append(f"  CAGR:           {om.get('cagr', 0):.2%}")
    lines.append(f"  Total Return:   {om.get('total_return', 0):.2%}")
    lines.append(f"  Max Drawdown:   {om.get('max_drawdown', 0):.2%}")
    lines.append(f"  Sharpe:         {om.get('sharpe', 0):.2f}")
    lines.append(f"  Calmar:         {om.get('calmar', 0):.2f}")
    lines.append(f"  Trading Days:   {om.get('n_days', 0)}")

    # IS/OOS degradation
    if is_cagrs and oos_cagrs:
        import numpy as np
        avg_is = np.mean(is_cagrs)
        avg_oos = np.mean(oos_cagrs)
        if avg_is > 0:
            degradation = 1.0 - avg_oos / avg_is
        else:
            degradation = float("nan")

        lines.append("")
        lines.append("-" * 90)
        lines.append("IS/OOS Degradation Analysis:")
        lines.append("-" * 90)
        lines.append(f"  Average IS CAGR:   {avg_is:>8.2%}")
        lines.append(f"  Average OOS CAGR:  {avg_oos:>8.2%}")
        lines.append(f"  Degradation:       {degradation:>8.1%}")
        if degradation <= 0.5:
            lines.append("  Verdict: ACCEPTABLE (< 50% degradation)")
        else:
            lines.append("  Verdict: HIGH DEGRADATION — possible overfitting")

    # Recommended live parameters
    fp = result.final_params
    lines.append("")
    lines.append("-" * 90)
    lines.append("Recommended Live Parameters (from final IS window):")
    lines.append("-" * 90)
    lines.append(f"  kama_period:     {fp.kama_period}")
    lines.append(f"  lookback_period: {fp.lookback_period}")
    lines.append(f"  kama_buffer:     {fp.kama_buffer}")
    lines.append(f"  top_n:           {fp.top_n}")

    lines.append("")
    lines.append("=" * 90)
    return "\n".join(lines)
```
## File: portfolio_sim/parallel.py
```python
"""Shared parallel evaluation infrastructure for Optuna-based searches.

Used by both optimizer.py (sensitivity analysis) and max_profit.py
(CAGR-maximizing search) to avoid duplicating worker initializers,
evaluation functions, and parameter suggestion logic.
"""

from __future__ import annotations

from typing import Callable

import optuna
import pandas as pd

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
    args: tuple[StrategyParams, float, Callable, str, list[str], list[str]],
) -> dict:
    """Evaluate a single param combo on the full dataset.

    Args:
        args: Tuple of (params, max_dd_limit, objective_fn, objective_key,
              param_keys, metric_keys).
            - params: strategy parameters to evaluate
            - max_dd_limit: passed to the objective function
            - objective_fn: callable(equity, max_dd_limit) -> float
            - objective_key: key name for the objective value in result dict
            - param_keys: param attribute names to include in result
            - metric_keys: metric dict keys to include in result

    Returns:
        Dict with parameter values, objective, and requested metrics.
    """
    params, max_dd_limit, objective_fn, objective_key, param_keys, metric_keys = args
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
```
## File: portfolio_sim/__main__.py
```python
"""Allow running as: python -m src.portfolio_sim <subcommand>."""

from src.portfolio_sim.cli import main

if __name__ == "__main__":
    main()
```
## File: portfolio_sim/data.py
```python
"""Data loading: tickers + price fetching via yfinance, Parquet cache.

Supports two universes:
  - S&P 500 (from local CSV or fallback URL)
  - Cross-asset ETFs (hardcoded in config.py)

Cache behavior:
  - If close_prices{suffix}.parquet and open_prices{suffix}.parquet exist in
    output/cache/, data is loaded from disk and yfinance is NOT called.
  - Use refresh=True to force re-download and overwrite cache.
"""

import pandas as pd
import structlog
import yfinance as yf
from pathlib import Path
from tqdm import tqdm

from src.portfolio_sim.config import CACHE_DIR, ETF_UNIVERSE, SPY_TICKER

log = structlog.get_logger(__name__)

CLOSE_CACHE = CACHE_DIR / "close_prices.parquet"
OPEN_CACHE = CACHE_DIR / "open_prices.parquet"


def fetch_etf_tickers() -> list[str]:
    """Return the hardcoded cross-asset ETF universe from config.

    No network call or CSV read needed. SPY is included as both
    a tradable asset and benchmark.
    """
    log.info("Using cross-asset ETF universe", n_tickers=len(ETF_UNIVERSE))
    return sorted(ETF_UNIVERSE)


def fetch_price_data(
    tickers: list[str],
    period: str = "5y",
    refresh: bool = False,
    cache_suffix: str = "",
    min_rows: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch Close and Open prices for all tickers.

    Returns (close_prices, open_prices) DataFrames with DatetimeIndex rows
    and ticker columns.

    Args:
        tickers: list of ticker symbols to download.
        period: yfinance period string (default: "5y").
        refresh: force re-download and overwrite cache.
        cache_suffix: appended to cache filenames to separate ETF vs S&P 500
                      caches (e.g. "_etf").
        min_rows: if > 0 and the cached data has fewer rows, automatically
                  re-download with the requested *period*.

    Cache: if output/cache/ contains the parquet files, loads from disk and
    does not download. Pass refresh=True to force re-download.
    """
    close_cache = CACHE_DIR / f"close_prices{cache_suffix}.parquet"
    open_cache = CACHE_DIR / f"open_prices{cache_suffix}.parquet"

    if close_cache.exists() and open_cache.exists() and not refresh:
        close_df = pd.read_parquet(close_cache)
        open_df = pd.read_parquet(open_cache)
        if min_rows and len(close_df) < min_rows:
            log.warning(
                "Cache has fewer rows than required, re-downloading",
                cached_rows=len(close_df),
                min_rows=min_rows,
                period=period,
            )
        else:
            log.info("Loading prices from Parquet cache (skip download)", suffix=cache_suffix)
            return close_df, open_df

    full_list = list(set(tickers + [SPY_TICKER]))
    log.info("Downloading prices via yfinance", n_tickers=len(full_list), period=period)

    close_df, open_df = _download_from_yfinance(full_list, period)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    close_df.to_parquet(close_cache)
    open_df.to_parquet(open_cache)
    log.info("Prices cached to Parquet — future runs will use cache", path=str(CACHE_DIR))

    return close_df, open_df


def _download_from_yfinance(
    tickers: list[str], period: str, batch_size: int = 100
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download OHLCV data via yfinance and extract Close/Open.

    Downloads in batches of batch_size to improve stability.
    """
    all_close = []
    all_open = []

    # Process in batches
    n_batches = (len(tickers) + batch_size - 1) // batch_size
    for batch_idx in tqdm(range(n_batches), desc="Downloading batches", unit="batch"):
        i = batch_idx * batch_size
        batch_tickers = tickers[i : i + batch_size]

        raw = yf.download(
            batch_tickers,
            period=period,
            group_by="ticker",
            auto_adjust=True,
            threads=True,
            progress=True,
        )

        if isinstance(raw.columns, pd.MultiIndex):
            close_batch = (
                raw.xs("Close", axis=1, level=1)
                if "Close" in raw.columns.get_level_values(1)
                else pd.DataFrame()
            )
            open_batch = (
                raw.xs("Open", axis=1, level=1)
                if "Open" in raw.columns.get_level_values(1)
                else pd.DataFrame()
            )
        else:
            # Case for single ticker
            close_batch = raw[["Close"]].rename(columns={"Close": batch_tickers[0]})
            open_batch = raw[["Open"]].rename(columns={"Open": batch_tickers[0]})

        all_close.append(close_batch)
        all_open.append(open_batch)

    # Combine results
    close_df = pd.concat(all_close, axis=1)
    open_df = pd.concat(all_open, axis=1)

    close_df = close_df.ffill().dropna(axis=1, how="all")
    open_df = open_df[close_df.columns].ffill()

    if close_df.index.tz is not None:
        close_df.index = close_df.index.tz_localize(None)
        open_df.index = open_df.index.tz_localize(None)

    log.info(
        "Download complete", tickers_received=len(close_df.columns), rows=len(close_df)
    )
    return close_df, open_df
```
## File: portfolio_sim/reporting.py
```python
"""Performance metrics, drawdown computation, and report generation."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.portfolio_sim.config import RISK_FREE_RATE


def compute_metrics(equity: pd.Series) -> dict:
    """Compute performance metrics for an equity curve.

    Returns dict with: total_return, cagr, max_drawdown, sharpe, calmar,
    annualized_vol, n_days.
    """
    if equity.empty or equity.iloc[0] <= 0:
        return {
            "total_return": 0.0,
            "cagr": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
            "calmar": 0.0,
            "annualized_vol": 0.0,
            "win_rate": 0.0,
            "n_days": 0,
        }

    days = len(equity)
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (252 / max(1, days)) - 1

    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    max_dd = abs(drawdown.min())

    returns = equity.pct_change().dropna()
    ann_vol = returns.std() * np.sqrt(252)

    sharpe = (cagr - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else 0.0
    calmar = cagr / max_dd if max_dd > 0 else 0.0

    win_rate = float((returns > 0).sum() / len(returns)) if len(returns) > 0 else 0.0

    return {
        "total_return": float(total_return),
        "cagr": float(cagr),
        "max_drawdown": float(max_dd),
        "sharpe": float(sharpe),
        "calmar": float(calmar),
        "annualized_vol": float(ann_vol),
        "win_rate": win_rate,
        "n_days": days,
    }


def compute_drawdown_series(equity: pd.Series) -> pd.Series:
    """Compute the underwater/drawdown series (values <= 0)."""
    if equity.empty:
        return pd.Series(dtype=float)
    rolling_max = equity.cummax()
    return (equity - rolling_max) / rolling_max


def compute_monthly_returns(equity: pd.Series) -> pd.DataFrame:
    """Return a Year x Month table of monthly returns (as fractions)."""
    monthly = equity.resample("ME").last().pct_change().dropna()
    table = monthly.groupby([monthly.index.year, monthly.index.month]).first().unstack()
    if isinstance(table.columns, pd.MultiIndex):
        table.columns = table.columns.droplevel(0)
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    table.columns = [month_names[c - 1] for c in table.columns]
    return table


def compute_yearly_returns(equity: pd.Series) -> pd.Series:
    """Return annual returns as a Series indexed by year."""
    yearly = equity.resample("YE").last()
    returns = yearly.pct_change().dropna()
    returns.index = returns.index.year
    return returns


def compute_rolling_sharpe(
    equity: pd.Series,
    window: int = 252,
    risk_free_rate: float = RISK_FREE_RATE,
) -> pd.Series:
    """Return rolling annualized Sharpe ratio."""
    returns = equity.pct_change().dropna()
    rolling_mean = returns.rolling(window).mean() * 252
    rolling_std = returns.rolling(window).std() * np.sqrt(252)
    sharpe = ((rolling_mean - risk_free_rate) / rolling_std).dropna()
    return sharpe


def format_metrics_table(metrics: dict) -> str:
    """Format metrics dict as a readable CLI table."""
    lines = [
        "Performance Metrics",
        "-" * 40,
        f"  Total Return:   {metrics['total_return']:>8.1%}",
        f"  CAGR:           {metrics['cagr']:>8.1%}",
        f"  Max Drawdown:   {metrics['max_drawdown']:>8.1%}",
        f"  Sharpe Ratio:   {metrics['sharpe']:>8.2f}",
        f"  Calmar Ratio:   {metrics['calmar']:>8.2f}",
        f"  Ann. Volatility:{metrics['annualized_vol']:>8.1%}",
        f"  Trading Days:   {metrics['n_days']:>8d}",
    ]
    return "\n".join(lines)


def format_comparison_table(strat_metrics: dict, spy_metrics: dict) -> str:
    """Format side-by-side comparison of strategy vs SPY metrics."""
    lines = [
        f"{'Metric':<20} {'Strategy':>12} {'S&P 500':>12}",
        "-" * 46,
        f"{'Total Return':<20} {strat_metrics['total_return']:>11.1%} {spy_metrics['total_return']:>11.1%}",
        f"{'CAGR':<20} {strat_metrics['cagr']:>11.1%} {spy_metrics['cagr']:>11.1%}",
        f"{'Max Drawdown':<20} {strat_metrics['max_drawdown']:>11.1%} {spy_metrics['max_drawdown']:>11.1%}",
        f"{'Sharpe Ratio':<20} {strat_metrics['sharpe']:>11.2f} {spy_metrics['sharpe']:>11.2f}",
        f"{'Calmar Ratio':<20} {strat_metrics['calmar']:>11.2f} {spy_metrics['calmar']:>11.2f}",
        f"{'Ann. Volatility':<20} {strat_metrics['annualized_vol']:>11.1%} {spy_metrics['annualized_vol']:>11.1%}",
        f"{'Trading Days':<20} {strat_metrics['n_days']:>11d} {spy_metrics['n_days']:>11d}",
    ]
    return "\n".join(lines)


def save_equity_png(
    equity: pd.Series,
    spy_equity: pd.Series,
    output_dir: Path,
    title: str = "KAMA Momentum Strategy vs S&P 500",
) -> Path:
    """Save strategy-vs-SPY equity comparison chart as PNG."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8),
        gridspec_kw={"height_ratios": [3, 1]}, sharex=True,
    )
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Panel 1: Equity curves
    ax1.plot(equity.index, equity.values, color="#2962FF", linewidth=1.5,
             label="KAMA Momentum")
    ax1.plot(spy_equity.index, spy_equity.values, color="#888888",
             linewidth=1.2, linestyle="--", label="S&P 500 (Buy & Hold)")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # Panel 2: Drawdown (strategy only)
    dd = compute_drawdown_series(equity)
    ax2.fill_between(dd.index, dd.values * 100, color="#e74c3c", alpha=0.5)
    ax2.plot(dd.index, dd.values * 100, color="#e74c3c", linewidth=0.8)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)

    # Metrics annotation for both
    strat_metrics = compute_metrics(equity)
    spy_metrics = compute_metrics(spy_equity)
    text = (
        f"Strategy -- CAGR: {strat_metrics['cagr']:.1%}  MaxDD: {strat_metrics['max_drawdown']:.1%}  "
        f"Sharpe: {strat_metrics['sharpe']:.2f}\n"
        f"S&P 500 -- CAGR: {spy_metrics['cagr']:.1%}  MaxDD: {spy_metrics['max_drawdown']:.1%}  "
        f"Sharpe: {spy_metrics['sharpe']:.2f}"
    )
    fig.text(0.5, 0.01, text, ha="center", fontsize=9, color="#555")

    fig.tight_layout(rect=[0, 0.05, 1, 0.97])

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "equity_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Equity curve saved to {path}")
    return path


def format_asset_report(
    sim_result,
    close_prices: pd.DataFrame,
    asset_meta: pd.DataFrame | None = None,
) -> str:
    """Format a report describing assets held by the strategy and their behavior.

    Args:
        sim_result: SimulationResult from engine.run_simulation.
        close_prices: Full close-price DataFrame (DatetimeIndex x tickers).
        asset_meta: Optional DataFrame with asset metadata (columns: Symbol,
            Shortname, Sector, Industry). Loaded from sp500_companies.csv.

    Returns:
        Human-readable text report.
    """
    holdings = sim_result.holdings_history  # DatetimeIndex x tickers, values = shares
    equity = sim_result.equity
    trade_log = sim_result.trade_log

    # Build metadata lookup
    meta_lookup: dict[str, dict] = {}
    if asset_meta is not None:
        for _, row in asset_meta.iterrows():
            meta_lookup[row["Symbol"]] = {
                "name": row.get("Shortname", ""),
                "sector": row.get("Sector", ""),
                "industry": row.get("Industry", ""),
            }

    # Identify tickers that were actually held (shares > 0 at least once)
    held_tickers = [
        t for t in holdings.columns
        if (holdings[t] > 0).any()
    ]

    if not held_tickers:
        return "No assets were held during the simulation period."

    # Trade counts per ticker
    buy_counts: Counter[str] = Counter()
    sell_counts: Counter[str] = Counter()
    for trade in trade_log:
        ticker = trade["ticker"]
        if trade["action"] == "buy":
            buy_counts[ticker] += 1
        else:
            sell_counts[ticker] += 1

    # Per-asset stats
    asset_rows: list[dict] = []
    sim_start = equity.index[0]
    sim_end = equity.index[-1]

    for ticker in held_tickers:
        shares = holdings[ticker]
        held_mask = shares > 0
        days_held = int(held_mask.sum())

        # Individual asset return over the full simulation window
        if ticker in close_prices.columns:
            px = close_prices[ticker].dropna()
            px_sim = px.loc[sim_start:sim_end]
            if len(px_sim) >= 2:
                asset_return = px_sim.iloc[-1] / px_sim.iloc[0] - 1
                asset_vol = px_sim.pct_change().dropna().std() * np.sqrt(252)
            else:
                asset_return = 0.0
                asset_vol = 0.0
        else:
            asset_return = 0.0
            asset_vol = 0.0

        # Average portfolio weight when held
        if days_held > 0 and ticker in close_prices.columns:
            px_aligned = close_prices[ticker].reindex(equity.index).ffill()
            position_value = shares * px_aligned
            weight = (position_value[held_mask] / equity[held_mask]).mean()
        else:
            weight = 0.0

        info = meta_lookup.get(ticker, {})
        asset_rows.append({
            "ticker": ticker,
            "name": info.get("name", ""),
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
            "return": asset_return,
            "volatility": asset_vol,
            "days_held": days_held,
            "buys": buy_counts.get(ticker, 0),
            "sells": sell_counts.get(ticker, 0),
            "avg_weight": weight,
        })

    # Sort by days held descending
    asset_rows.sort(key=lambda r: r["days_held"], reverse=True)

    # --- Format report ---
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("ASSET REPORT")
    lines.append("=" * 70)

    # Summary
    total_days = len(equity)
    avg_hold = np.mean([r["days_held"] for r in asset_rows])
    sector_counts = Counter(r["sector"] for r in asset_rows if r["sector"])
    top_sectors = sector_counts.most_common(5)

    lines.append("")
    lines.append(f"Simulation period: {sim_start.strftime('%Y-%m-%d')} — "
                 f"{sim_end.strftime('%Y-%m-%d')} ({total_days} trading days)")
    lines.append(f"Unique assets traded: {len(asset_rows)}")
    lines.append(f"Average holding period: {avg_hold:.0f} days")
    lines.append("")
    lines.append("Top sectors:")
    for sector, cnt in top_sectors:
        lines.append(f"  {sector:<30s} {cnt} assets")

    # Per-asset table
    lines.append("")
    lines.append("-" * 70)
    lines.append("Per-Asset Breakdown:")
    lines.append("-" * 70)

    for row in asset_rows:
        ticker = row["ticker"]
        name = row["name"]
        header = f"{ticker}" + (f" — {name}" if name else "")
        lines.append(f"\n  {header}")
        if row["sector"]:
            lines.append(f"    Sector:     {row['sector']}")
        if row["industry"]:
            lines.append(f"    Industry:   {row['industry']}")
        lines.append(f"    Return:     {row['return']:>+8.1%}   "
                     f"Volatility: {row['volatility']:>7.1%}")
        lines.append(f"    Days held:  {row['days_held']:>5d} / {total_days}   "
                     f"Avg weight: {row['avg_weight']:>6.1%}")
        lines.append(f"    Trades:     {row['buys']} buys, {row['sells']} sells")

    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)
```
## File: portfolio_sim/__pycache__/config.cpython-314.pyc
```
Error reading src/portfolio_sim/__pycache__/config.cpython-314.pyc: 'utf-8' codec can't decode byte 0x95 in position 8: invalid start byte
```
## File: portfolio_sim/__pycache__/optimization.cpython-314.pyc
```
Error reading src/portfolio_sim/__pycache__/optimization.cpython-314.pyc: 'utf-8' codec can't decode byte 0xa0 in position 9: invalid start byte
```
## File: portfolio_sim/__pycache__/data.cpython-314.pyc
```
Error reading src/portfolio_sim/__pycache__/data.cpython-314.pyc: 'utf-8' codec can't decode byte 0x9f in position 10: invalid start byte
```
## File: portfolio_sim/__pycache__/cli_utils.cpython-314.pyc
```
Error reading src/portfolio_sim/__pycache__/cli_utils.cpython-314.pyc: 'utf-8' codec can't decode byte 0xef in position 8: invalid continuation byte
```
## File: portfolio_sim/__pycache__/indicators._kama_recurrent_loop-11.py314.nbi
```
Error reading src/portfolio_sim/__pycache__/indicators._kama_recurrent_loop-11.py314.nbi: 'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte
```
## File: portfolio_sim/__pycache__/reporting.cpython-314.pyc
```
Error reading src/portfolio_sim/__pycache__/reporting.cpython-314.pyc: 'utf-8' codec can't decode bytes in position 9-10: invalid continuation byte
```
## File: portfolio_sim/__pycache__/indicators.cpython-314.pyc
```
Error reading src/portfolio_sim/__pycache__/indicators.cpython-314.pyc: 'utf-8' codec can't decode byte 0xaa in position 8: invalid start byte
```
## File: portfolio_sim/__pycache__/optimizer.cpython-314.pyc
```
Error reading src/portfolio_sim/__pycache__/optimizer.cpython-314.pyc: 'utf-8' codec can't decode byte 0x9f in position 10: invalid start byte
```
## File: portfolio_sim/__pycache__/params.cpython-314.pyc
```
Error reading src/portfolio_sim/__pycache__/params.cpython-314.pyc: 'utf-8' codec can't decode byte 0xcd in position 8: invalid continuation byte
```
## File: portfolio_sim/__pycache__/cli.cpython-314.pyc
```
Error reading src/portfolio_sim/__pycache__/cli.cpython-314.pyc: 'utf-8' codec can't decode byte 0x9f in position 10: invalid start byte
```
## File: portfolio_sim/__pycache__/models.cpython-314.pyc
```
Error reading src/portfolio_sim/__pycache__/models.cpython-314.pyc: 'utf-8' codec can't decode byte 0x83 in position 8: invalid start byte
```
## File: portfolio_sim/__pycache__/parallel.cpython-314.pyc
```
Error reading src/portfolio_sim/__pycache__/parallel.cpython-314.pyc: 'utf-8' codec can't decode byte 0x9f in position 10: invalid start byte
```
## File: portfolio_sim/__pycache__/__init__.cpython-314.pyc
```
Error reading src/portfolio_sim/__pycache__/__init__.cpython-314.pyc: 'utf-8' codec can't decode byte 0xe3 in position 16: invalid continuation byte
```
## File: portfolio_sim/__pycache__/__main__.cpython-314.pyc
```
Error reading src/portfolio_sim/__pycache__/__main__.cpython-314.pyc: 'utf-8' codec can't decode byte 0xef in position 8: invalid continuation byte
```
## File: portfolio_sim/__pycache__/alpha.cpython-314.pyc
```
Error reading src/portfolio_sim/__pycache__/alpha.cpython-314.pyc: 'utf-8' codec can't decode byte 0x82 in position 12: invalid start byte
```
## File: portfolio_sim/__pycache__/indicators._kama_recurrent_loop-11.py314.1.nbc
```
Error reading src/portfolio_sim/__pycache__/indicators._kama_recurrent_loop-11.py314.1.nbc: 'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte
```
## File: portfolio_sim/__pycache__/max_profit.cpython-314.pyc
```
Error reading src/portfolio_sim/__pycache__/max_profit.cpython-314.pyc: 'utf-8' codec can't decode byte 0x9f in position 10: invalid start byte
```
## File: portfolio_sim/__pycache__/engine.cpython-314.pyc
```
Error reading src/portfolio_sim/__pycache__/engine.cpython-314.pyc: 'utf-8' codec can't decode byte 0x9f in position 10: invalid start byte
```
## File: portfolio_sim/__pycache__/walk_forward.cpython-314.pyc
```
Error reading src/portfolio_sim/__pycache__/walk_forward.cpython-314.pyc: 'utf-8' codec can't decode byte 0x9f in position 10: invalid start byte
```
## File: portfolio_sim/commands/max_profit.py
```python
"""Maximum profit parameter search (TPE or Pareto NSGA-II)."""

from src.portfolio_sim.cli_utils import (
    create_output_dir,
    filter_valid_tickers,
    setup_logging,
)
from src.portfolio_sim.config import INITIAL_CAPITAL
from src.portfolio_sim.data import fetch_etf_tickers, fetch_price_data
from src.portfolio_sim.engine import run_simulation
from src.portfolio_sim.max_profit import (
    format_max_profit_report,
    format_pareto_report,
    run_max_profit_pareto,
    run_max_profit_search,
    select_best_from_pareto,
)
from src.portfolio_sim.params import StrategyParams
from src.portfolio_sim.reporting import compute_metrics, format_comparison_table

COMMAND_NAME = "max-profit"


def register(subparsers) -> None:
    p = subparsers.add_parser(
        COMMAND_NAME, help="Maximum profit parameter search (TPE or Pareto NSGA-II)",
    )
    p.add_argument(
        "--refresh", action="store_true",
        help="Force refresh data cache from yfinance",
    )
    p.add_argument(
        "--period", default="3y",
        help="yfinance period string (default: 3y)",
    )
    p.add_argument(
        "--n-workers", type=int, default=None,
        help="Number of parallel workers (default: cpu_count - 1)",
    )
    p.add_argument(
        "--max-dd", type=float, default=0.60,
        help="Max drawdown rejection limit (default: 0.60 = 60%%)",
    )
    p.add_argument(
        "--n-trials", type=int, default=50,
        help="Number of Optuna trials per universe (default: 50)",
    )
    p.add_argument(
        "--pareto", action="store_true",
        help="Use multi-objective Pareto search (NSGA-II) instead of single-objective TPE",
    )


def _run_verification(close_prices, open_prices, tickers, initial_capital, params, universe_name):
    """Run simulation with given params and print metrics."""
    print(f"\n{'=' * 70}")
    print(f"VERIFICATION — {universe_name} (default params)")
    print(f"{'=' * 70}")

    result = run_simulation(
        close_prices, open_prices, tickers, initial_capital,
        params=params, show_progress=True,
    )

    strat_metrics = compute_metrics(result.equity)
    spy_metrics = compute_metrics(result.spy_equity)

    print(f"\n{format_comparison_table(strat_metrics, spy_metrics)}")
    return strat_metrics


def run(args) -> None:
    setup_logging()
    output_dir = create_output_dir("max_profit")

    summary_lines: list[str] = []
    summary_lines.append("=" * 90)
    summary_lines.append("MAXIMUM PROFIT SEARCH — SUMMARY")
    summary_lines.append(f"Period: {args.period}  |  Max DD limit: {args.max_dd:.0%}")
    summary_lines.append("=" * 90)

    # ETF
    print("\nFetching ETF universe...")
    etf_tickers = fetch_etf_tickers()
    print(f"Universe: {len(etf_tickers)} tickers")

    print(f"Downloading price data ({args.period})...")
    close_etf, open_etf = fetch_price_data(
        etf_tickers, period=args.period, refresh=args.refresh,
        cache_suffix="_etf",
    )

    etf_params = StrategyParams(
        use_risk_adjusted=True,
        enable_regime_filter=False,
        enable_correlation_filter=True,
        sizing_mode="risk_parity",
    )
    min_days = etf_params.warmup * 2
    valid_etf = filter_valid_tickers(close_etf, min_days)
    print(f"Tradable tickers with {min_days}+ days: {len(valid_etf)}")

    _run_verification(
        close_etf, open_etf, valid_etf, INITIAL_CAPITAL,
        etf_params, "Cross-Asset ETF",
    )

    fixed = {
        "enable_correlation_filter": True,
        "correlation_threshold": 0.65,
        "correlation_lookback": 60,
        "use_risk_adjusted": True,
        "sizing_mode": "risk_parity",
    }

    if args.pareto:
        print(f"\n{'=' * 70}")
        print(f"PARETO SEARCH (NSGA-II) — Cross-Asset ETF ({args.n_trials} trials)")
        print(f"{'=' * 70}\n")

        etf_result = run_max_profit_pareto(
            close_etf, open_etf, valid_etf, INITIAL_CAPITAL,
            universe="etf",
            default_params=etf_params,
            fixed_params=fixed,
            n_trials=args.n_trials,
            n_workers=args.n_workers,
        )

        report_etf = format_pareto_report(etf_result)
        best_pareto = select_best_from_pareto(etf_result)
        if best_pareto:
            print(f"\nBest from Pareto front (by Calmar):")
            print(f"  kama={best_pareto.kama_period}, lookback={best_pareto.lookback_period}, "
                  f"buffer={best_pareto.kama_buffer}, top_n={best_pareto.top_n}")

        if etf_result.pareto_front is not None:
            pareto_path = output_dir / "pareto_front_etf.csv"
            etf_result.pareto_front.to_csv(pareto_path, index=False)
            print(f"Pareto front saved to {pareto_path}")
    else:
        print(f"\n{'=' * 70}")
        print(f"OPTUNA SEARCH — Cross-Asset ETF ({args.n_trials} trials)")
        print(f"{'=' * 70}\n")

        etf_result = run_max_profit_search(
            close_etf, open_etf, valid_etf, INITIAL_CAPITAL,
            universe="etf",
            default_params=etf_params,
            fixed_params=fixed,
            n_trials=args.n_trials,
            n_workers=args.n_workers,
            max_dd_limit=args.max_dd,
        )

        report_etf = format_max_profit_report(etf_result)

    print(f"\n{report_etf}")

    etf_result.grid_results.to_csv(
        output_dir / "grid_results_etf.csv", index=False,
    )
    (output_dir / "report_etf.txt").write_text(report_etf)
    summary_lines.append("")
    summary_lines.append(report_etf)

    summary = "\n".join(summary_lines)
    (output_dir / "summary.txt").write_text(summary)
    print(f"\nAll results saved to {output_dir}")
```
## File: portfolio_sim/commands/optimize.py
```python
"""Parameter sensitivity analysis (Optuna TPE)."""

import sys

from src.portfolio_sim.cli_utils import (
    create_output_dir,
    filter_valid_tickers,
    setup_logging,
)
from src.portfolio_sim.config import INITIAL_CAPITAL
from src.portfolio_sim.data import fetch_etf_tickers, fetch_price_data
from src.portfolio_sim.engine import run_simulation
from src.portfolio_sim.optimizer import (
    format_sensitivity_report,
    run_sensitivity,
)
from src.portfolio_sim.params import StrategyParams
from src.portfolio_sim.reporting import format_asset_report, save_equity_png

COMMAND_NAME = "optimize"


def register(subparsers) -> None:
    p = subparsers.add_parser(
        COMMAND_NAME, help="Parameter sensitivity analysis (Optuna TPE)",
    )
    p.add_argument(
        "--refresh", action="store_true",
        help="Force refresh data cache from yfinance",
    )
    p.add_argument(
        "--period", default="3y",
        help="yfinance period string (default: 3y)",
    )
    p.add_argument(
        "--n-workers", type=int, default=None,
        help="Number of parallel workers (default: cpu_count - 1)",
    )
    p.add_argument(
        "--n-trials", type=int, default=50,
        help="Number of Optuna trials for sensitivity analysis (default: 50)",
    )


def run(args) -> None:
    setup_logging()
    output_dir = create_output_dir("sens")

    print("\nUsing cross-asset ETF universe...")
    tickers = fetch_etf_tickers()
    print(f"Universe: {len(tickers)} tickers")

    min_days = 504
    print(f"Downloading price data ({args.period})...")
    close_prices, open_prices = fetch_price_data(
        tickers, period=args.period, refresh=args.refresh,
        cache_suffix="_etf", min_rows=min_days,
    )

    valid_tickers = filter_valid_tickers(close_prices, min_days)
    print(f"Tradable tickers with {min_days}+ days: {len(valid_tickers)}")

    if not valid_tickers:
        print(f"\nERROR: No tickers with {min_days}+ trading days.")
        print("Try: python -m src.portfolio_sim optimize --refresh")
        sys.exit(1)

    base_params = StrategyParams()
    print(f"\nStarting sensitivity analysis (Optuna TPE, {args.n_trials} trials)...")
    print(f"  Base params: kama={base_params.kama_period}, "
          f"lookback={base_params.lookback_period}, "
          f"buffer={base_params.kama_buffer}, top_n={base_params.top_n}")
    print()

    result = run_sensitivity(
        close_prices,
        open_prices,
        valid_tickers,
        INITIAL_CAPITAL,
        base_params=base_params,
        n_trials=args.n_trials,
        n_workers=args.n_workers,
    )

    report = format_sensitivity_report(result)
    print(f"\n{report}")

    report_path = output_dir / "sensitivity_report.txt"
    report_path.write_text(report)
    print(f"\nReport saved to {report_path}")

    grid_path = output_dir / "trial_results.csv"
    result.grid_results.to_csv(grid_path, index=False)
    print(f"Trial results saved to {grid_path}")

    print("\nRunning base-params simulation for detailed report...")
    sim_result = run_simulation(
        close_prices, open_prices, valid_tickers, INITIAL_CAPITAL,
        params=base_params,
    )

    save_equity_png(sim_result.equity, sim_result.spy_equity, output_dir)

    asset_report = format_asset_report(sim_result, close_prices, asset_meta=None)
    asset_report_path = output_dir / "asset_report.txt"
    asset_report_path.write_text(asset_report)
    print(f"Asset report saved to {asset_report_path}")
```
## File: portfolio_sim/commands/walk_forward.py
```python
"""Walk-forward optimization."""

import sys

import pandas as pd

from src.portfolio_sim.cli_utils import (
    create_output_dir,
    filter_valid_tickers,
    setup_logging,
)
from src.portfolio_sim.config import INITIAL_CAPITAL
from src.portfolio_sim.data import fetch_etf_tickers, fetch_price_data
from src.portfolio_sim.params import StrategyParams
from src.portfolio_sim.walk_forward import format_wfo_report, run_walk_forward

COMMAND_NAME = "walk-forward"


def register(subparsers) -> None:
    p = subparsers.add_parser(
        COMMAND_NAME, help="Walk-forward optimization",
    )
    p.add_argument(
        "--refresh", action="store_true",
        help="Force refresh data cache from yfinance",
    )
    p.add_argument(
        "--period", default="3y",
        help="yfinance period string (default: 3y)",
    )
    p.add_argument(
        "--n-workers", type=int, default=None,
        help="Number of parallel workers (default: cpu_count - 1)",
    )
    p.add_argument(
        "--n-trials", type=int, default=50,
        help="Number of Optuna trials per WFO step (default: 50)",
    )
    p.add_argument(
        "--oos-days", type=int, default=126,
        help="OOS window size in trading days (default: 126 ~ 6 months)",
    )
    p.add_argument(
        "--min-is-days", type=int, default=378,
        help="Minimum IS window size in trading days (default: 378 ~ 1.5 years)",
    )


def run(args) -> None:
    setup_logging()
    output_dir = create_output_dir("wfo")

    print("\nUsing cross-asset ETF universe...")
    tickers = fetch_etf_tickers()
    print(f"Universe: {len(tickers)} tickers")

    min_days = args.min_is_days
    print(f"Downloading price data ({args.period})...")
    close_prices, open_prices = fetch_price_data(
        tickers, period=args.period, refresh=args.refresh,
        cache_suffix="_etf", min_rows=min_days,
    )

    valid_tickers = filter_valid_tickers(close_prices, min_days)
    print(f"Tradable tickers with {min_days}+ days: {len(valid_tickers)}")

    if not valid_tickers:
        print(f"\nERROR: No tickers with {min_days}+ trading days.")
        print("Try: python -m src.portfolio_sim walk-forward --refresh")
        sys.exit(1)

    base_params = StrategyParams()
    print(f"\nStarting walk-forward optimization...")
    print(f"  IS minimum: {args.min_is_days} days (~{args.min_is_days / 252:.0f} years)")
    print(f"  OOS window: {args.oos_days} days (~{args.oos_days / 252:.0f} years)")
    print(f"  Trials per step: {args.n_trials}")
    print()

    result = run_walk_forward(
        close_prices,
        open_prices,
        valid_tickers,
        INITIAL_CAPITAL,
        base_params=base_params,
        n_trials_per_step=args.n_trials,
        n_workers=args.n_workers,
        min_is_days=args.min_is_days,
        oos_days=args.oos_days,
    )

    report = format_wfo_report(result)
    print(f"\n{report}")

    report_path = output_dir / "wfo_report.txt"
    report_path.write_text(report)
    print(f"\nReport saved to {report_path}")

    equity_path = output_dir / "stitched_oos_equity.csv"
    result.stitched_equity.to_csv(equity_path, header=True)
    print(f"Stitched OOS equity saved to {equity_path}")

    step_rows = []
    for step in result.steps:
        step_rows.append({
            "step": step.step_index + 1,
            "is_start": step.is_start.date(),
            "is_end": step.is_end.date(),
            "oos_start": step.oos_start.date(),
            "oos_end": step.oos_end.date(),
            "kama_period": step.optimized_params.kama_period,
            "lookback_period": step.optimized_params.lookback_period,
            "kama_buffer": step.optimized_params.kama_buffer,
            "top_n": step.optimized_params.top_n,
            "is_cagr": step.is_metrics.get("cagr", 0),
            "is_maxdd": step.is_metrics.get("max_drawdown", 0),
            "oos_cagr": step.oos_metrics.get("cagr", 0),
            "oos_maxdd": step.oos_metrics.get("max_drawdown", 0),
            "oos_sharpe": step.oos_metrics.get("sharpe", 0),
        })

    steps_df = pd.DataFrame(step_rows)
    steps_path = output_dir / "wfo_steps.csv"
    steps_df.to_csv(steps_path, index=False)
    print(f"Step details saved to {steps_path}")

    fp = result.final_params
    print(f"\nRecommended live parameters:")
    print(f"  kama_period={fp.kama_period}, lookback_period={fp.lookback_period}, "
          f"kama_buffer={fp.kama_buffer}, top_n={fp.top_n}")
```
## File: portfolio_sim/commands/__pycache__/simulate.cpython-314.pyc
```
Error reading src/portfolio_sim/commands/__pycache__/simulate.cpython-314.pyc: 'utf-8' codec can't decode byte 0xb5 in position 8: invalid start byte
```
## File: portfolio_sim/commands/__pycache__/backtest.cpython-314.pyc
```
Error reading src/portfolio_sim/commands/__pycache__/backtest.cpython-314.pyc: 'utf-8' codec can't decode byte 0xf0 in position 8: invalid continuation byte
```
## File: portfolio_sim/commands/__pycache__/optimize.cpython-314.pyc
```
Error reading src/portfolio_sim/commands/__pycache__/optimize.cpython-314.pyc: 'utf-8' codec can't decode byte 0x9f in position 10: invalid start byte
```
## File: portfolio_sim/commands/__pycache__/__init__.cpython-314.pyc
```
Error reading src/portfolio_sim/commands/__pycache__/__init__.cpython-314.pyc: 'utf-8' codec can't decode byte 0x94 in position 8: invalid start byte
```
## File: portfolio_sim/commands/__pycache__/max_profit.cpython-314.pyc
```
Error reading src/portfolio_sim/commands/__pycache__/max_profit.cpython-314.pyc: 'utf-8' codec can't decode byte 0x9f in position 10: invalid start byte
```
## File: portfolio_sim/commands/__pycache__/walk_forward.cpython-314.pyc
```
Error reading src/portfolio_sim/commands/__pycache__/walk_forward.cpython-314.pyc: 'utf-8' codec can't decode byte 0x9f in position 10: invalid start byte
```
