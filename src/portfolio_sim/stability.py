"""One-at-a-time (OAT) parameter stability analysis via Walk-Forward Optimization.

For each parameter in the search space, fixes it at each candidate value,
runs a full WFO with the remaining parameters optimised, and records
OOS metrics.  This reveals which parameters sit on stable plateaus
vs. fragile cliffs.
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import structlog

from src.portfolio_sim.config import (
    INITIAL_CAPITAL,
    MAX_DD_LIMIT,
    SEARCH_SPACE,
    get_kama_periods,
)
from src.portfolio_sim.optimizer import precompute_kama_caches
from src.portfolio_sim.parallel import init_eval_worker
from src.portfolio_sim.params import StrategyParams
from src.portfolio_sim.walk_forward import run_walk_forward

log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

OOS_METRIC_KEYS = ("sharpe", "cagr", "max_drawdown", "calmar", "total_return", "annualized_vol")


@dataclass
class ParamStabilityPoint:
    """Result of one WFO run with a single parameter fixed."""

    param_name: str
    param_value: float | int
    oos_metrics: dict
    n_wfo_steps: int
    failed: bool = False


@dataclass
class ParamStabilityResult:
    """Stability analysis for one parameter across all candidate values."""

    param_name: str
    base_value: float | int
    points: list[ParamStabilityPoint]

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for pt in self.points:
            row = {"param_value": pt.param_value, "failed": pt.failed, "n_wfo_steps": pt.n_wfo_steps}
            for k in OOS_METRIC_KEYS:
                row[f"oos_{k}"] = pt.oos_metrics.get(k, float("nan"))
            rows.append(row)
        return pd.DataFrame(rows)


@dataclass
class StabilityAnalysisResult:
    """Complete stability analysis across all tested parameters."""

    param_results: dict[str, ParamStabilityResult]
    base_params: StrategyParams

    def summary_dataframe(self) -> pd.DataFrame:
        rows = []
        for name, pr in self.param_results.items():
            for pt in pr.points:
                row = {"param_name": name, "param_value": pt.param_value, "failed": pt.failed}
                for k in OOS_METRIC_KEYS:
                    row[f"oos_{k}"] = pt.oos_metrics.get(k, float("nan"))
                rows.append(row)
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def generate_candidate_values(
    param_name: str,
    space: dict[str, dict],
    max_points: int | None = None,
) -> list[float | int]:
    """Generate discrete candidate values for a parameter from its search space spec."""
    spec = space[param_name]
    ptype = spec["type"]

    if ptype == "categorical":
        values = list(spec["choices"])
    elif ptype == "int":
        values = list(range(spec["low"], spec["high"] + 1, spec.get("step", 1)))
    elif ptype == "float":
        step = spec["step"]
        values = [round(v, 6) for v in np.arange(spec["low"], spec["high"] + step / 2, step)]
    else:
        raise ValueError(f"Unknown param type: {ptype}")

    if max_points is not None and len(values) > max_points >= 2:
        indices = np.round(np.linspace(0, len(values) - 1, max_points)).astype(int)
        values = [values[i] for i in indices]

    return values


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def run_param_stability(
    close_prices: pd.DataFrame,
    open_prices: pd.DataFrame,
    tickers: list[str],
    initial_capital: float = INITIAL_CAPITAL,
    base_params: StrategyParams | None = None,
    space: dict[str, dict] | None = None,
    param_names: list[str] | None = None,
    n_trials_per_step: int = 50,
    n_workers: int | None = None,
    min_is_days: int | None = None,
    oos_days: int | None = None,
    max_dd_limit: float = MAX_DD_LIMIT,
    metric: str = "sharpe",
    max_points_per_param: int | None = None,
    high_prices: pd.DataFrame | None = None,
    low_prices: pd.DataFrame | None = None,
) -> StabilityAnalysisResult:
    """Run OAT parameter stability analysis via walk-forward optimization."""
    base_params = base_params or StrategyParams()
    space = space or SEARCH_SPACE
    param_names = param_names or list(space.keys())
    if not n_workers or n_workers < 1:
        n_workers = os.cpu_count()

    # Count total runs for progress
    total_runs = 0
    for pn in param_names:
        total_runs += len(generate_candidate_values(pn, space, max_points_per_param))

    log.info(
        "stability_start",
        n_params=len(param_names),
        total_wfo_runs=total_runs,
        n_trials_per_step=n_trials_per_step,
    )

    # Pre-compute KAMA caches once for all periods in the full space
    all_kama_periods = get_kama_periods(space)
    kama_caches = precompute_kama_caches(close_prices, tickers, all_kama_periods, n_workers)
    log.info("stability_kama_precomputed", n_periods=len(kama_caches))

    # Create a single shared executor
    executor = ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=init_eval_worker,
        initargs=(close_prices, open_prices, tickers, initial_capital, kama_caches,
                  high_prices, low_prices),
    )

    param_results: dict[str, ParamStabilityResult] = {}
    run_idx = 0

    try:
        for param_name in param_names:
            candidates = generate_candidate_values(param_name, space, max_points_per_param)
            base_val = getattr(base_params, param_name)
            points: list[ParamStabilityPoint] = []

            for val_idx, value in enumerate(candidates):
                run_idx += 1
                log.info(
                    "stability_wfo_start",
                    param=param_name,
                    value=value,
                    run=f"{run_idx}/{total_runs}",
                )

                fixed_params = replace(base_params, **{param_name: value})
                reduced_space = {k: v for k, v in space.items() if k != param_name}

                try:
                    wfo_result = run_walk_forward(
                        close_prices, open_prices, tickers,
                        initial_capital=initial_capital,
                        base_params=fixed_params,
                        space=reduced_space,
                        n_trials_per_step=n_trials_per_step,
                        n_workers=n_workers,
                        min_is_days=min_is_days,
                        oos_days=oos_days,
                        max_dd_limit=max_dd_limit,
                        metric=metric,
                        kama_caches=kama_caches,
                        executor=executor,
                        verbose=False,
                        high_prices=high_prices,
                        low_prices=low_prices,
                    )
                    point = ParamStabilityPoint(
                        param_name=param_name,
                        param_value=value,
                        oos_metrics=wfo_result.oos_metrics,
                        n_wfo_steps=len(wfo_result.steps),
                    )
                    log.info(
                        "stability_wfo_done",
                        param=param_name,
                        value=value,
                        oos_sharpe=f"{wfo_result.oos_metrics.get('sharpe', 0):.2f}",
                        oos_cagr=f"{wfo_result.oos_metrics.get('cagr', 0):.2%}",
                    )
                except Exception:
                    log.warning("stability_wfo_failed", param=param_name, value=value, exc_info=True)
                    point = ParamStabilityPoint(
                        param_name=param_name,
                        param_value=value,
                        oos_metrics={},
                        n_wfo_steps=0,
                        failed=True,
                    )
                points.append(point)

            param_results[param_name] = ParamStabilityResult(
                param_name=param_name,
                base_value=base_val,
                points=points,
            )
    finally:
        executor.shutdown(wait=True)

    log.info("stability_done", n_params=len(param_results))
    return StabilityAnalysisResult(param_results=param_results, base_params=base_params)


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def _compute_cv(values: list[float]) -> float:
    """Coefficient of variation (std / |mean|). Returns NaN if insufficient data."""
    clean = [v for v in values if math.isfinite(v)]
    if len(clean) < 2:
        return float("nan")
    mean = np.mean(clean)
    if abs(mean) < 1e-9:
        return float("nan")
    return float(np.std(clean) / abs(mean))


def _verdict(cv: float) -> str:
    if math.isnan(cv):
        return "N/A"
    if cv < 0.10:
        return "STABLE"
    if cv < 0.25:
        return "MODERATE"
    return "FRAGILE"


def format_stability_report(result: StabilityAnalysisResult, metric: str = "sharpe") -> str:
    """Format a human-readable parameter stability report."""
    metric_key = f"oos_{metric}" if not metric.startswith("oos_") else metric
    raw_metric = metric_key.removeprefix("oos_")

    lines: list[str] = []
    lines.append("=" * 90)
    lines.append("PARAMETER STABILITY ANALYSIS (OAT via Walk-Forward Optimization)")
    lines.append("=" * 90)

    bp = result.base_params
    lines.append("")
    lines.append(
        f"Base: r2_window={bp.r2_window}, kama_asset_period={bp.kama_asset_period}, "
        f"kama_buffer={bp.kama_buffer}, top_n={bp.top_n}, rebal_days={bp.rebal_days}"
    )
    lines.append(
        f"      max_per_class={bp.max_per_class}, target_vol={bp.target_vol}, "
        f"max_leverage={bp.max_leverage}, portfolio_vol_lookback={bp.portfolio_vol_lookback}, "
        f"min_invested_pct={bp.min_invested_pct}"
    )
    lines.append(f"Metric: {raw_metric}")

    summary_rows: list[tuple[str, float, str]] = []

    for param_name, pr in result.param_results.items():
        df = pr.to_dataframe()
        valid = df[~df["failed"]]

        lines.append("")
        lines.append("-" * 90)
        lines.append(f"Parameter: {param_name} (base={pr.base_value})")
        lines.append("-" * 90)

        header = f"  {'Value':>10}  {'OOS Sharpe':>11}  {'OOS CAGR':>9}  {'OOS MaxDD':>10}  {'OOS Calmar':>11}"
        lines.append(header)

        for _, row in df.iterrows():
            if row["failed"]:
                lines.append(f"  {row['param_value']:>10}  {'FAILED':>11}")
                continue
            marker = "  \u25c4 base" if row["param_value"] == pr.base_value else ""
            lines.append(
                f"  {row['param_value']:>10}  "
                f"{row['oos_sharpe']:>11.2f}  "
                f"{row['oos_cagr']:>8.1%}  "
                f"{row['oos_max_drawdown']:>9.1%}  "
                f"{row['oos_calmar']:>11.2f}"
                f"{marker}"
            )

        metric_values = valid[metric_key].tolist() if metric_key in valid.columns else []
        cv = _compute_cv(metric_values)
        v = _verdict(cv)
        if math.isnan(cv):
            lines.append(f"  CV: N/A   Verdict: {v}")
        else:
            lines.append(f"  CV: {cv:.1%}   Verdict: {v}")
        summary_rows.append((param_name, cv, v))

    lines.append("")
    lines.append("=" * 90)
    lines.append(f"SUMMARY (by OOS {raw_metric})")
    lines.append("=" * 90)
    lines.append(f"  {'Parameter':<25} {'CV%':>8}  {'Verdict':<10}")
    lines.append("  " + "-" * 45)
    for name, cv, v in summary_rows:
        cv_str = f"{cv:.1%}" if math.isfinite(cv) else "N/A"
        lines.append(f"  {name:<25} {cv_str:>8}  {v:<10}")

    lines.append("")
    lines.append("=" * 90)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def save_stability_csvs(result: StabilityAnalysisResult, output_dir: Path) -> None:
    """Save per-parameter and summary CSVs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Summary CSV
    summary = result.summary_dataframe()
    summary_path = output_dir / "stability_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Summary CSV saved to {summary_path}")

    # Per-parameter CSVs
    per_param_dir = output_dir / "per_param"
    per_param_dir.mkdir(parents=True, exist_ok=True)
    for name, pr in result.param_results.items():
        df = pr.to_dataframe()
        path = per_param_dir / f"stability_{name}.csv"
        df.to_csv(path, index=False)

    print(f"Per-parameter CSVs saved to {per_param_dir}/")


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def save_stability_charts(
    result: StabilityAnalysisResult,
    output_dir: Path,
    metric: str = "sharpe",
) -> None:
    """Save per-parameter stability charts and an overview grid."""
    metric_key = f"oos_{metric}" if not metric.startswith("oos_") else metric
    raw_metric = metric_key.removeprefix("oos_")

    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    param_names = list(result.param_results.keys())

    # Individual charts
    for name in param_names:
        pr = result.param_results[name]
        df = pr.to_dataframe()
        valid = df[~df["failed"]]
        if valid.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(valid["param_value"], valid[metric_key], "o-", color="#2962FF", linewidth=2, markersize=6)
        ax.axvline(pr.base_value, color="#e74c3c", linestyle="--", linewidth=1, alpha=0.8, label=f"base={pr.base_value}")

        ax.set_xlabel(name, fontsize=11)
        ax.set_ylabel(f"OOS {raw_metric}", fontsize=11)
        ax.set_title(f"Stability: {name}", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(charts_dir / f"stability_{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Overview grid
    n = len(param_names)
    cols = min(3, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, name in enumerate(param_names):
        ax = axes[idx]
        pr = result.param_results[name]
        df = pr.to_dataframe()
        valid = df[~df["failed"]]
        if valid.empty:
            ax.set_title(name)
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        ax.plot(valid["param_value"], valid[metric_key], "o-", color="#2962FF", linewidth=1.5, markersize=4)
        ax.axvline(pr.base_value, color="#e74c3c", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.set_ylabel(f"OOS {raw_metric}", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f"Parameter Stability Overview (OOS {raw_metric})", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    overview_path = charts_dir / "stability_overview.png"
    fig.savefig(overview_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Stability charts saved to {charts_dir}/")
