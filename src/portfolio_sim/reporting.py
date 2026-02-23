"""Performance metrics, drawdown computation, and report generation."""

import json
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

    return {
        "total_return": float(total_return),
        "cagr": float(cagr),
        "max_drawdown": float(max_dd),
        "sharpe": float(sharpe),
        "calmar": float(calmar),
        "annualized_vol": float(ann_vol),
        "n_days": days,
    }


def compute_drawdown_series(equity: pd.Series) -> pd.Series:
    """Compute the underwater/drawdown series (values <= 0)."""
    if equity.empty:
        return pd.Series(dtype=float)
    rolling_max = equity.cummax()
    return (equity - rolling_max) / rolling_max


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


def save_equity_png(
    equity: pd.Series, output_dir: Path, title: str = "Equity Curve"
) -> Path:
    """Save equity curve + drawdown chart as PNG. Returns path."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Equity curve
    ax1.plot(equity.index, equity.values, color="#2962FF", linewidth=1.2)
    ax1.fill_between(equity.index, equity.values, alpha=0.08, color="#2962FF")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # Drawdown
    dd = compute_drawdown_series(equity)
    ax2.fill_between(dd.index, dd.values * 100, color="#e74c3c", alpha=0.5)
    ax2.plot(dd.index, dd.values * 100, color="#e74c3c", linewidth=0.8)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    # Metrics annotation
    metrics = compute_metrics(equity)
    text = (
        f"CAGR: {metrics['cagr']:.1%}  |  "
        f"MaxDD: {metrics['max_drawdown']:.1%}  |  "
        f"Sharpe: {metrics['sharpe']:.2f}  |  "
        f"Calmar: {metrics['calmar']:.2f}"
    )
    fig.text(0.5, 0.01, text, ha="center", fontsize=10, color="#555")

    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "equity_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Equity curve saved to {path}")
    return path


def save_wfv_report(wfv_result: dict, metric: str, output_dir: Path) -> Path:
    """Save Walk-Forward Validation markdown report. Returns path."""
    oos_equity = wfv_result["oos_equity"]
    windows = wfv_result["windows"]
    oos_metrics = compute_metrics(oos_equity)

    report = [
        "# Walk-Forward Validation Report",
        f"\n**OOS Period:** {oos_equity.index[0].strftime('%Y-%m-%d')} to "
        f"{oos_equity.index[-1].strftime('%Y-%m-%d')}",
        f"**Optimization Metric:** {metric.upper()}",
        f"**Windows:** {len(windows)}",
        "\n## OOS Performance (Blind Test)",
        f"\n- **Total Return:** {oos_metrics['total_return']:.1%}",
        f"- **CAGR:** {oos_metrics['cagr']:.1%}",
        f"- **Max Drawdown:** {oos_metrics['max_drawdown']:.1%}",
        f"- **Sharpe:** {oos_metrics['sharpe']:.2f}",
        f"- **Calmar:** {oos_metrics['calmar']:.2f}",
        "\n## Window Breakdown",
        "\n| Window | Train Period | Test Period | IS Score | OOS Return | OOS MaxDD |",
        "| :---: | :--- | :--- | :---: | :---: | :---: |",
    ]

    for w in windows:
        report.append(
            f"| {w['window']} | {w['train_start']} -> {w['train_end']} | "
            f"{w['test_start']} -> {w['test_end']} | "
            f"{w['is_score']:.4f} | {w['oos_return_pct']:.1f}% | "
            f"{w['oos_max_dd_pct']:.1f}% |"
        )

    report.append("\n## Parameters Per Window")
    for w in windows:
        report.append(f"\n### Window {w['window']}")
        report.append("\n| Parameter | Value |")
        report.append("| :--- | :--- |")
        for k, v in w["params"].items():
            report.append(f"| {k} | {v} |")

    report.append("\n---")
    report.append(
        f"\n*Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "wfv_report.md"
    path.write_text("\n".join(report))
    print(f"WFV report saved to {path}")
    return path


def save_wfv_json(wfv_result: dict, output_dir: Path) -> Path:
    """Save WFV window details as JSON. Returns path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "wfv_windows.json"
    with open(path, "w") as f:
        json.dump(wfv_result["windows"], f, indent=2, default=str)
    print(f"WFV JSON saved to {path}")
    return path
