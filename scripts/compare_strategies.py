"""Compare R² Momentum, MVO benchmarks, and SPY side-by-side.

Usage:
    uv run python scripts/compare_strategies.py
    uv run python scripts/compare_strategies.py --period 5y
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.portfolio_sim.benchmark_mvo.engine import run_mvo_backtest
from src.portfolio_sim.benchmark_mvo.params import MVOParams
from src.portfolio_sim.cli_utils import create_output_dir, filter_valid_tickers, setup_logging
from src.portfolio_sim.config import INITIAL_CAPITAL, SPY_TICKER
from src.portfolio_sim.data import fetch_etf_tickers, fetch_price_data
from src.portfolio_sim.engine import run_backtest
from src.portfolio_sim.reporting import compute_drawdown_series, compute_metrics


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

METRICS_ORDER = [
    ("Total Return", "total_return", ".1%"),
    ("CAGR", "cagr", ".1%"),
    ("Max Drawdown", "max_drawdown", ".1%"),
    ("Sharpe Ratio", "sharpe", ".2f"),
    ("Calmar Ratio", "calmar", ".2f"),
    ("Ann. Volatility", "annualized_vol", ".1%"),
    ("Win Rate", "win_rate", ".1%"),
    ("Trading Days", "n_days", "d"),
]


def format_multi_table(results: dict[str, dict]) -> str:
    """Format N-strategy comparison table."""
    names = list(results.keys())
    col_w = max(14, max(len(n) + 2 for n in names))
    header = f"{'Metric':<20}" + "".join(f"{n:>{col_w}}" for n in names)
    sep = "-" * len(header)

    lines = [header, sep]
    for label, key, fmt in METRICS_ORDER:
        row = f"{label:<20}"
        for name in names:
            val = results[name].get(key, 0)
            row += f"{val:>{col_w}{fmt}}"
        lines.append(row)
    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

COLORS = {
    "R2 Momentum": "#2962FF",
    "MVO MinVar": "#FF6D00",
    "MVO MaxSharpe": "#AA00FF",
    "MVO RiskParity": "#00C853",
    "SPY": "#888888",
}


def plot_comparison(
    curves: dict[str, pd.Series],
    output_dir: Path,
) -> None:
    """Overlay equity curves and drawdowns for all strategies."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 9),
        gridspec_kw={"height_ratios": [2, 1]}, sharex=True,
    )
    fig.suptitle("Strategy Comparison", fontsize=14, fontweight="bold")

    for name, eq in curves.items():
        color = COLORS.get(name, "#000000")
        ls = "--" if name == "SPY" else "-"
        lw = 1.2 if name == "SPY" else 1.5

        ax1.plot(eq.index, eq.values, color=color, linewidth=lw,
                 linestyle=ls, label=name)

        dd = compute_drawdown_series(eq)
        ax2.plot(dd.index, dd.values * 100, color=color, linewidth=0.8,
                 linestyle=ls, label=name)

    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.legend(loc="lower left", fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "strategy_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Compare all strategies")
    parser.add_argument("--period", default="3y")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    setup_logging()

    print("=" * 60)
    print("Multi-Strategy Comparison")
    print("=" * 60)

    # Load data once (shared across all strategies)
    print("\nLoading data...")
    tickers = fetch_etf_tickers()
    close_prices, open_prices = fetch_price_data(
        tickers, period=args.period, refresh=args.refresh, cache_suffix="_etf",
    )

    # Use MVO warmup (262) as common minimum to align equity curves
    min_days = 262
    valid_tickers = [
        t for t in filter_valid_tickers(close_prices, min_days)
        if t != SPY_TICKER
    ]
    print(f"  Universe: {len(valid_tickers)} tickers, "
          f"{close_prices.index[0].date()} .. {close_prices.index[-1].date()}")

    if not valid_tickers:
        print("ERROR: No valid tickers.")
        return

    results: dict[str, dict] = {}
    curves: dict[str, pd.Series] = {}

    # 1. R² Momentum (V1)
    print("\n[1/4] Running R² Momentum...")
    r2_eq, spy_eq, _, _, r2_trades = run_backtest(
        close_prices, open_prices, valid_tickers,
        initial_capital=INITIAL_CAPITAL,
        warmup_days=min_days,
    )
    results["R2 Momentum"] = compute_metrics(r2_eq)
    results["SPY"] = compute_metrics(spy_eq)
    curves["R2 Momentum"] = r2_eq
    curves["SPY"] = spy_eq
    print(f"       {len(r2_trades)} trades")

    # 2. MVO Min-Variance
    print("[2/4] Running MVO Min-Variance...")
    mv_eq, _, _, _, mv_trades = run_mvo_backtest(
        close_prices, open_prices, valid_tickers,
        initial_capital=INITIAL_CAPITAL,
        params=MVOParams(objective="min_variance"),
    )
    results["MVO MinVar"] = compute_metrics(mv_eq)
    curves["MVO MinVar"] = mv_eq
    print(f"       {len(mv_trades)} trades")

    # 3. MVO Max-Sharpe
    print("[3/4] Running MVO Max-Sharpe...")
    ms_eq, _, _, _, ms_trades = run_mvo_backtest(
        close_prices, open_prices, valid_tickers,
        initial_capital=INITIAL_CAPITAL,
        params=MVOParams(objective="max_sharpe"),
    )
    results["MVO MaxSharpe"] = compute_metrics(ms_eq)
    curves["MVO MaxSharpe"] = ms_eq
    print(f"       {len(ms_trades)} trades")

    # 4. MVO Risk Parity
    print("[4/4] Running MVO Risk Parity...")
    rp_eq, _, _, _, rp_trades = run_mvo_backtest(
        close_prices, open_prices, valid_tickers,
        initial_capital=INITIAL_CAPITAL,
        params=MVOParams(objective="risk_parity"),
    )
    results["MVO RiskParity"] = compute_metrics(rp_eq)
    curves["MVO RiskParity"] = rp_eq
    print(f"       {len(rp_trades)} trades")

    # Print comparison table
    print("\n" + format_multi_table(results))

    # Save outputs
    output_dir = create_output_dir("compare_strategies")
    plot_comparison(curves, output_dir)
    pd.DataFrame(results).T.to_csv(output_dir / "metrics.csv")
    print(f"  Metrics saved: {output_dir / 'metrics.csv'}")
    print(f"\nAll outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
