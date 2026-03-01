"""Visualization and standalone backtest runner for R² Momentum strategy."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.portfolio_sim.config import (
    ATR_PERIOD,
    GAP_THRESHOLD,
    INITIAL_CAPITAL,
    KAMA_BUFFER,
    KAMA_PERIOD,
    KAMA_SPY_PERIOD,
    R2_LOOKBACK,
    REBAL_PERIOD_WEEKS,
    TOP_N,
)
from src.portfolio_sim.engine import load_data, run_backtest
from src.portfolio_sim.reporting import compute_drawdown_series, compute_metrics

OUTPUT_DIR = Path("output/r2_momentum")

COLOR_STRATEGY = "#2962FF"
COLOR_SPY = "#888888"
COLOR_DD = "#D32F2F"


def plot_equity_and_drawdown(
    equity: pd.Series,
    spy_equity: pd.Series,
    output_dir: Path,
) -> None:
    """2-panel chart: equity curves (top) + drawdown (bottom)."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [2, 1]}, sharex=True,
    )
    fig.suptitle(
        "Clenow R² Momentum Strategy", fontsize=14, fontweight="bold",
    )

    ax1.plot(
        equity.index, equity.values,
        color=COLOR_STRATEGY, linewidth=1.5, label="R² Momentum",
    )
    ax1.plot(
        spy_equity.index, spy_equity.values,
        color=COLOR_SPY, linewidth=1.2, linestyle="--", label="SPY (Buy & Hold)",
    )
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    m = compute_metrics(equity)
    m_spy = compute_metrics(spy_equity)
    annotation = (
        f"R² Mom: CAGR {m['cagr']:.1%}  Sharpe {m['sharpe']:.2f}  "
        f"MaxDD {m['max_drawdown']:.1%}  |  "
        f"SPY: CAGR {m_spy['cagr']:.1%}  Sharpe {m_spy['sharpe']:.2f}  "
        f"MaxDD {m_spy['max_drawdown']:.1%}"
    )
    ax1.text(
        0.5, 0.02, annotation, transform=ax1.transAxes,
        ha="center", fontsize=8, color="#555",
    )

    dd = compute_drawdown_series(equity)
    dd_spy = compute_drawdown_series(spy_equity)
    ax2.fill_between(dd.index, dd.values * 100, color=COLOR_DD, alpha=0.3)
    ax2.plot(dd.index, dd.values * 100, color=COLOR_DD, linewidth=0.8, label="R² Momentum")
    ax2.plot(
        dd_spy.index, dd_spy.values * 100,
        color=COLOR_SPY, linewidth=0.8, linestyle="--", label="SPY",
    )
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.legend(loc="lower left", fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "equity_and_drawdown.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_holdings_over_time(
    selection_history: list[tuple[pd.Timestamp, list[str]]],
    output_dir: Path,
) -> None:
    """Plot number of holdings over time."""
    if not selection_history:
        return

    dates = [d for d, _ in selection_history]
    counts = [len(tickers) for _, tickers in selection_history]

    fig, ax = plt.subplots(figsize=(14, 4))
    fig.suptitle("Number of Holdings Over Time", fontsize=14, fontweight="bold")

    ax.step(dates, counts, color=COLOR_STRATEGY, linewidth=1.2, where="post")
    ax.fill_between(dates, counts, step="post", color=COLOR_STRATEGY, alpha=0.15)
    ax.set_ylabel("Positions")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(counts) + 2)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = output_dir / "holdings_over_time.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_turnover(
    trade_log: list[dict],
    output_dir: Path,
) -> None:
    """Plot monthly trade count."""
    if not trade_log:
        return

    df = pd.DataFrame(trade_log)
    df["month"] = df["date"].dt.to_period("M")
    monthly = df.groupby("month").size()

    fig, ax = plt.subplots(figsize=(14, 4))
    fig.suptitle("Monthly Trade Count", fontsize=14, fontweight="bold")

    x = [p.to_timestamp() for p in monthly.index]
    ax.bar(x, monthly.values, width=20, color=COLOR_STRATEGY, alpha=0.7)
    ax.set_ylabel("Number of Trades")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = output_dir / "monthly_turnover.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_asset_frequency(
    selection_history: list[tuple[pd.Timestamp, list[str]]],
    output_dir: Path,
    top_k: int = 20,
) -> None:
    """Horizontal bar chart of most frequently held assets."""
    if not selection_history:
        return

    freq: Counter[str] = Counter()
    for _, selected in selection_history:
        for t in selected:
            freq[t] += 1

    if not freq:
        return

    most_common = freq.most_common(top_k)
    tickers = [t for t, _ in reversed(most_common)]
    counts = [c for _, c in reversed(most_common)]

    fig, ax = plt.subplots(figsize=(10, max(6, len(tickers) * 0.35)))
    fig.suptitle(
        f"Top {top_k} Most Frequently Held Assets", fontsize=14, fontweight="bold",
    )

    ax.barh(tickers, counts, color=COLOR_STRATEGY, alpha=0.7)
    ax.set_xlabel("Times Selected")
    ax.grid(True, alpha=0.3, axis="x")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = output_dir / "asset_frequency.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def main() -> None:
    """Run Clenow R² Momentum strategy backtest."""
    from src.portfolio_sim.engine import format_report

    print("=" * 60)
    print("Clenow R² Momentum Strategy")
    print("=" * 60)

    print("\nLoading data...")
    close_prices, open_prices, tickers = load_data(period="3y")
    print(f"  Universe: {len(tickers)} tickers")
    print(f"  Date range: {close_prices.index[0].date()} to {close_prices.index[-1].date()}")
    print(f"  Lookback: {R2_LOOKBACK}d, KAMA asset: {KAMA_PERIOD}d, KAMA SPY: {KAMA_SPY_PERIOD}d, buffer: {KAMA_BUFFER}")
    print(f"  Gap filter: {GAP_THRESHOLD:.0%}, ATR period: {ATR_PERIOD}d")
    print(f"  Rebalance: every {REBAL_PERIOD_WEEKS} weeks (incremental lazy-hold)")

    print("\nRunning R² Momentum backtest...")
    equity, spy_equity, _hh, _ch, trade_log = run_backtest(
        close_prices, open_prices, tickers,
        initial_capital=INITIAL_CAPITAL, top_n=TOP_N,
        rebal_period_weeks=REBAL_PERIOD_WEEKS,
    )
    m = compute_metrics(equity)
    print(f"  CAGR: {m['cagr']:.1%}  Sharpe: {m['sharpe']:.2f}  MaxDD: {m['max_drawdown']:.1%}")
    print(f"  Total trades: {len(trade_log)}")

    report = format_report(equity, spy_equity)
    print(f"\n{report}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "report.txt").write_text(report, encoding="utf-8")

    equity_df = pd.DataFrame({"R2_Momentum": equity, "SPY": spy_equity})
    equity_df.to_csv(OUTPUT_DIR / "equity_curves.csv")

    if trade_log:
        pd.DataFrame(trade_log).to_csv(OUTPUT_DIR / "trade_log.csv", index=False)

    print("\nGenerating plots...")
    plot_equity_and_drawdown(equity, spy_equity, OUTPUT_DIR)
    plot_turnover(trade_log, OUTPUT_DIR)

    print(f"\nAll outputs saved to {OUTPUT_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
