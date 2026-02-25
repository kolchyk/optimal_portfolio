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
