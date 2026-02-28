"""Clenow R² Momentum Strategy — ETF rotation backtest.

Based on Andreas Clenow's "Stocks on the Move" methodology:
  - R² Momentum scoring: annualized regression slope × R²
  - SPY KAMA regime filter (market-level)
  - Individual asset KAMA trend filter
  - Gap filter (>15% single-day move exclusion)
  - ATR-based risk parity position sizing
  - Lazy-hold incremental rebalancing (low turnover)

Usage:
    uv run python scripts/compare_methods.py
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.portfolio_sim.config import (
    COMMISSION_RATE,
    INITIAL_CAPITAL,
    KAMA_BUFFER,
    KAMA_PERIOD,
    KAMA_SPY_PERIOD,
    RISK_FREE_RATE,
    SLIPPAGE_RATE,
    SPY_TICKER,
    TOP_N,
)
from src.portfolio_sim.data import fetch_etf_tickers, fetch_price_data
from src.portfolio_sim.indicators import compute_kama_series
from src.portfolio_sim.reporting import compute_drawdown_series, compute_metrics

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COST_RATE = COMMISSION_RATE + SLIPPAGE_RATE
DAILY_RF = (1 + RISK_FREE_RATE) ** (1 / 252) - 1

# Clenow R² Momentum parameters
R2_LOOKBACK = 90           # 90 trading days (Clenow standard)
KAMA_ASSET = KAMA_PERIOD   # KAMA period for individual asset trend filter
KAMA_SPY = KAMA_SPY_PERIOD # KAMA period for SPY regime filter
KAMA_BUF = KAMA_BUFFER     # hysteresis buffer for KAMA filters
GAP_THRESHOLD = 0.15       # exclude assets with >15% single-day gap
ATR_PERIOD = 20            # ATR lookback for position sizing
RISK_FACTOR = 0.001        # 10 bps risk per position per day (Clenow default)

REBAL_PERIOD_WEEKS = 2     # rebalance check every 2 weeks (~10 trading days)

OUTPUT_DIR = Path("output/r2_momentum")

# Plot colors
COLOR_STRATEGY = "#2962FF"   # blue
COLOR_SPY = "#888888"        # gray
COLOR_DD = "#D32F2F"         # red for drawdowns


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(
    period: str = "3y", refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load ETF data using existing data.py utilities."""
    tickers = fetch_etf_tickers()
    close_prices, open_prices = fetch_price_data(
        tickers, period=period, refresh=refresh, cache_suffix="_etf",
    )

    min_days = max(R2_LOOKBACK, KAMA_SPY, KAMA_ASSET) + 10
    valid_tickers = [
        t for t in close_prices.columns
        if t != SPY_TICKER and close_prices[t].dropna().shape[0] >= min_days
    ]

    return close_prices, open_prices, valid_tickers


# ---------------------------------------------------------------------------
# Filters & utilities
# ---------------------------------------------------------------------------
def is_risk_on(
    spy_kama: pd.Series,
    spy_close: pd.Series,
    date: pd.Timestamp,
    kama_buffer: float = KAMA_BUF,
) -> bool:
    """Return True if SPY close > KAMA * (1 - buffer) — bull regime."""
    kama_val = spy_kama.get(date, np.nan)
    if np.isnan(kama_val):
        return True  # not enough data yet, assume risk-on
    close_val = spy_close.get(date, np.nan)
    if np.isnan(close_val):
        return True
    return float(close_val) > kama_val * (1 - kama_buffer)


def is_above_kama(
    kama_series: pd.Series,
    prices: pd.Series,
    date: pd.Timestamp,
    kama_buffer: float = KAMA_BUF,
) -> bool:
    """Return True if asset's close > KAMA * (1 - buffer) on given date."""
    kama_val = kama_series.get(date, np.nan)
    if np.isnan(kama_val):
        return True  # not enough data, assume above
    close_val = prices.get(date, np.nan)
    if np.isnan(close_val):
        return True
    return float(close_val) > kama_val * (1 - kama_buffer)


def has_large_gap(
    prices: pd.Series,
    date: pd.Timestamp,
    lookback: int = R2_LOOKBACK,
    threshold: float = GAP_THRESHOLD,
) -> bool:
    """Return True if any single-day return exceeds threshold within lookback.

    Detects earnings gaps, splits, and other dislocations that distort
    the regression-based momentum score.
    """
    prices_up_to = prices.loc[:date].dropna()
    if len(prices_up_to) < lookback:
        return False
    window = prices_up_to.iloc[-lookback:]
    daily_returns = window.pct_change().dropna().abs()
    return bool((daily_returns > threshold).any())


def compute_atr_close(
    prices: pd.Series, date: pd.Timestamp, period: int = ATR_PERIOD,
) -> float:
    """Compute ATR approximation using absolute close-to-close changes.

    True ATR requires High/Low data. This approximation uses
    the rolling mean of |close_t - close_{t-1}| which works well
    for liquid ETFs where intraday ranges are moderate.
    """
    prices_up_to = prices.loc[:date].dropna()
    if len(prices_up_to) < period + 1:
        return np.nan
    abs_changes = prices_up_to.diff().abs().iloc[-period:]
    return float(abs_changes.mean())


# ---------------------------------------------------------------------------
# R² Momentum scoring
# ---------------------------------------------------------------------------
def compute_r2_momentum(
    prices: pd.Series, period: int = R2_LOOKBACK,
) -> tuple[float, float, float]:
    """Fit OLS to log-prices, return (annualized_return, r_squared, score).

    score = annualized_return × r_squared (Clenow's adjusted momentum).
    Period: 90 trading days (Clenow standard).
    """
    if len(prices) < period:
        return (np.nan, np.nan, np.nan)

    window = prices.iloc[-period:]
    log_p = np.log(window.values)

    if np.any(np.isnan(log_p)) or np.any(np.isinf(log_p)):
        return (np.nan, np.nan, np.nan)

    x = np.arange(period, dtype=float)
    slope, intercept = np.polyfit(x, log_p, 1)

    annualized_return = float(np.exp(slope * 252) - 1)

    y_hat = slope * x + intercept
    ss_res = float(np.sum((log_p - y_hat) ** 2))
    ss_tot = float(np.sum((log_p - log_p.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0
    r_squared = max(0.0, min(1.0, r_squared))

    score = annualized_return * r_squared
    return (annualized_return, r_squared, score)


# ---------------------------------------------------------------------------
# Asset selection with Clenow filters + ATR risk parity
# ---------------------------------------------------------------------------
def select_r2_assets(
    close_prices: pd.DataFrame,
    date: pd.Timestamp,
    eligible_tickers: list[str],
    top_n: int,
    kama_cache: dict[str, pd.Series],
    spy_kama: pd.Series,
    kama_buffer: float = KAMA_BUF,
    r2_lookback: int = R2_LOOKBACK,
    gap_threshold: float = GAP_THRESHOLD,
    atr_period: int = ATR_PERIOD,
    risk_factor: float = RISK_FACTOR,
) -> tuple[dict[str, float], dict[str, float]]:
    """Select top-N assets by Clenow R² Momentum, weight by ATR risk parity.

    Returns:
        (target_weights, all_scores) where:
        - target_weights: {ticker: weight} for desired portfolio (sum~=1.0)
        - all_scores: {ticker: r2_momentum_score} for all passing assets
          (used by the engine for exit decisions on held positions)

    Filter cascade:
      1. SPY > KAMA (market regime)
      2. Asset > KAMA (individual trend)
      3. No gap > 15% in last 90 days
      4. R² Momentum score > 0
    Sizing: ATR-based risk parity (normalized to sum=1.0, no leverage)
    """
    if not is_risk_on(spy_kama, close_prices[SPY_TICKER], date, kama_buffer):
        return {}, {}

    prices_up_to = close_prices.loc[:date]
    scores: dict[str, float] = {}

    for t in eligible_tickers:
        if t not in prices_up_to.columns:
            continue
        ser = prices_up_to[t].dropna()

        # Filter: asset above KAMA
        ticker_kama = kama_cache.get(t, pd.Series(dtype=float))
        if not is_above_kama(ticker_kama, ser, date, kama_buffer):
            continue

        # Filter: no large gaps
        if has_large_gap(ser, date, lookback=r2_lookback, threshold=gap_threshold):
            continue

        # Score: R² Momentum
        _, _, score = compute_r2_momentum(ser, period=r2_lookback)
        if not np.isnan(score) and score > 0:
            scores[t] = score

    if not scores:
        return {}, scores

    # Rank and take top_n
    ranked = sorted(scores, key=scores.get, reverse=True)[:top_n]

    # ATR-based risk parity position sizing
    atr_inv: dict[str, float] = {}
    for t in ranked:
        ser = prices_up_to[t].dropna()
        atr = compute_atr_close(ser, date, period=atr_period)
        price = float(ser.iloc[-1])
        if np.isnan(atr) or atr < 1e-10 or price < 1e-10:
            atr_inv[t] = 1.0  # fallback
        else:
            # Weight proportional to RISK_FACTOR / (ATR / price)
            # Low ATR% assets get larger positions (risk parity)
            atr_pct = atr / price
            atr_inv[t] = risk_factor / atr_pct

    total = sum(atr_inv.values())
    if total < 1e-10:
        return {}, scores
    weights = {t: v / total for t, v in atr_inv.items()}

    return weights, scores


# ---------------------------------------------------------------------------
# Backtest engine (lazy-hold incremental rebalancing)
# ---------------------------------------------------------------------------
def run_backtest(
    close_prices: pd.DataFrame,
    open_prices: pd.DataFrame,
    tickers: list[str],
    initial_capital: float = INITIAL_CAPITAL,
    top_n: int = TOP_N,
    warmup_days: int | None = None,
    rebal_period_weeks: int = REBAL_PERIOD_WEEKS,
    kama_asset_period: int = KAMA_ASSET,
    kama_spy_period: int = KAMA_SPY,
    kama_buffer: float = KAMA_BUF,
    r2_lookback: int = R2_LOOKBACK,
    gap_threshold: float = GAP_THRESHOLD,
    atr_period: int = ATR_PERIOD,
    risk_factor: float = RISK_FACTOR,
    kama_cache_ext: dict[str, pd.Series] | None = None,
    spy_kama_ext: pd.Series | None = None,
) -> tuple[pd.Series, list[tuple[pd.Timestamp, list[str]]], list[dict]]:
    """Run Clenow R² Momentum backtest with KAMA filters and incremental rebalancing.

    Lazy-hold approach (following engine.py pattern):
    - SELL only when: KAMA breach, failed filters, negative score, or SPY regime off
    - BUY only to fill empty slots from top-ranked new candidates
    - Positions that lost rank but maintain their trend are kept

    Returns:
        (equity_series, selection_history, trade_log)
    """
    if warmup_days is None:
        warmup_days = max(r2_lookback, kama_spy_period, kama_asset_period) + 10

    sim_dates = close_prices.index[warmup_days:]
    if len(sim_dates) == 0:
        return pd.Series(dtype=float), [], []

    # Use pre-computed KAMA if provided, else compute
    if kama_cache_ext is not None and spy_kama_ext is not None:
        kama_cache = kama_cache_ext
        spy_kama = spy_kama_ext
    else:
        kama_cache: dict[str, pd.Series] = {}
        for t in tickers:
            if t in close_prices.columns:
                kama_cache[t] = compute_kama_series(
                    close_prices[t].dropna(), period=kama_asset_period,
                )
        spy_kama = compute_kama_series(
            close_prices[SPY_TICKER].dropna(), period=kama_spy_period,
        )

    # Rebalance-check dates: every N weeks (~N*5 trading days)
    rebal_interval = rebal_period_weeks * 5
    rebalance_dates = set(sim_dates[::rebal_interval])
    rebalance_dates.add(sim_dates[0])  # ensure initial portfolio entry

    cash = initial_capital
    shares: dict[str, float] = {}
    equity_values: list[float] = []
    selection_history: list[tuple[pd.Timestamp, list[str]]] = []
    trade_log: list[dict] = []

    # Pending trades: computed on Close(T), executed on Open(T+1)
    pending_sells: list[str] | None = None
    pending_buys: dict[str, float] | None = None  # {ticker: weight} for new buys
    execute_on_next_open = False

    for date in sim_dates:
        daily_close = close_prices.loc[date]
        daily_open = open_prices.loc[date]

        # --- Execute pending trades on Open ---
        if execute_on_next_open:
            equity_at_open = cash + sum(
                shares[t] * daily_open.get(t, 0.0) for t in shares
            )
            if equity_at_open <= 0:
                equity_at_open = max(cash, 1.0)

            total_cost = 0.0

            # 1. Execute sells first
            if pending_sells:
                for t in pending_sells:
                    if t in shares:
                        price = daily_open.get(t, 0.0)
                        if price > 0 and shares[t] > 0:
                            trade_value = shares[t] * price
                            total_cost += trade_value * COST_RATE
                            trade_log.append({
                                "date": date, "ticker": t, "action": "sell",
                                "shares": shares[t], "price": price,
                            })
                        del shares[t]

            # 2. Compute available cash for buys
            held_value = sum(
                shares[t] * daily_open.get(t, 0.0) for t in shares
            )
            available = equity_at_open - held_value - total_cost

            # 3. Execute buys with available cash
            if pending_buys and available > 0:
                available_for_buys = available
                for t, w in pending_buys.items():
                    price = daily_open.get(t, np.nan)
                    if np.isnan(price) or price <= 0:
                        continue
                    max_allocation = available_for_buys * w
                    allocation = min(max_allocation, available)
                    if allocation > 0:
                        net_investment = allocation / (1 + COST_RATE)
                        cost = net_investment * COST_RATE
                        shares[t] = net_investment / price
                        total_cost += cost
                        available -= allocation
                        trade_log.append({
                            "date": date, "ticker": t, "action": "buy",
                            "shares": shares[t], "price": price,
                        })

            cash = equity_at_open - sum(
                shares[t] * daily_open.get(t, 0.0) for t in shares
            ) - total_cost

            execute_on_next_open = False
            pending_sells = None
            pending_buys = None

        # --- Cash earns risk-free rate ---
        if cash > 0:
            cash *= 1 + DAILY_RF

        # --- Mark-to-market on Close ---
        equity = cash + sum(
            shares.get(t, 0) * daily_close.get(t, 0.0) for t in shares
        )
        equity_values.append(equity)

        if equity <= 0:
            remaining = len(sim_dates) - len(equity_values)
            equity_values.extend([0.0] * remaining)
            break

        # --- On rebalance-check date: compute signals for lazy-hold ---
        if date in rebalance_dates:
            target_weights, all_scores = select_r2_assets(
                close_prices, date, tickers, top_n,
                kama_cache=kama_cache, spy_kama=spy_kama,
                kama_buffer=kama_buffer,
                r2_lookback=r2_lookback,
                gap_threshold=gap_threshold,
                atr_period=atr_period,
                risk_factor=risk_factor,
            )

            # Determine sells: positions that should be exited
            sells_to_do: list[str] = []
            prices_up_to = close_prices.loc[:date]

            for t in list(shares.keys()):
                should_sell = False

                # Exit 1: SPY regime off (target_weights empty = risk-off)
                if not target_weights and not all_scores:
                    should_sell = True
                # Exit 2: asset dropped below KAMA
                elif t in close_prices.columns:
                    ser = prices_up_to[t].dropna()
                    ticker_kama = kama_cache.get(t, pd.Series(dtype=float))
                    if not is_above_kama(ticker_kama, ser, date, kama_buffer):
                        should_sell = True
                    # Exit 3: R² Momentum score turned negative or asset
                    # failed hard filters (not in all_scores)
                    elif t not in all_scores:
                        should_sell = True

                if should_sell:
                    sells_to_do.append(t)

            # Determine buys: fill empty slots from top-ranked candidates
            kept_positions = set(shares.keys()) - set(sells_to_do)
            open_slots = top_n - len(kept_positions)

            buys_to_do: dict[str, float] = {}
            if open_slots > 0 and target_weights:
                for t in sorted(
                    target_weights,
                    key=lambda x: all_scores.get(x, 0),
                    reverse=True,
                ):
                    if t not in shares:
                        buys_to_do[t] = target_weights[t]
                        open_slots -= 1
                        if open_slots <= 0:
                            break

                # Re-normalize buy weights to sum = 1.0
                if buys_to_do:
                    buy_total = sum(buys_to_do.values())
                    if buy_total > 0:
                        buys_to_do = {
                            t: v / buy_total for t, v in buys_to_do.items()
                        }

            if sells_to_do or buys_to_do:
                pending_sells = sells_to_do if sells_to_do else None
                pending_buys = buys_to_do if buys_to_do else None
                execute_on_next_open = True
                final_holdings = list(kept_positions | set(buys_to_do.keys()))
                selection_history.append((date, final_holdings))

    n = len(equity_values)
    return (
        pd.Series(equity_values, index=sim_dates[:n]),
        selection_history,
        trade_log,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def format_report(equity: pd.Series, spy_equity: pd.Series) -> str:
    """Format single-strategy report: Clenow R² Momentum vs SPY."""
    strat_m = compute_metrics(equity)
    spy_m = compute_metrics(spy_equity)

    header = f"{'Metric':<20} {'R² Momentum':>16} {'SPY':>16}"
    sep = "-" * len(header)

    rows = [
        ("Total Return", "total_return", ".1%"),
        ("CAGR", "cagr", ".1%"),
        ("Max Drawdown", "max_drawdown", ".1%"),
        ("Sharpe Ratio", "sharpe", ".2f"),
        ("Calmar Ratio", "calmar", ".2f"),
        ("Ann. Volatility", "annualized_vol", ".1%"),
        ("Win Rate", "win_rate", ".1%"),
        ("Trading Days", "n_days", "d"),
    ]

    lines = [
        "=" * len(header),
        "Clenow R² Momentum Strategy Report",
        "=" * len(header),
        "",
        f"Parameters: lookback={R2_LOOKBACK}d, KAMA_asset={KAMA_ASSET}d, "
        f"KAMA_SPY={KAMA_SPY}d, KAMA_buffer={KAMA_BUF}, gap={GAP_THRESHOLD:.0%}, "
        f"ATR={ATR_PERIOD}d, top_n={TOP_N}, rebal={REBAL_PERIOD_WEEKS}w",
        "",
        header,
        sep,
    ]

    for label, key, fmt in rows:
        row = f"{label:<20}{strat_m[key]:>16{fmt}}{spy_m[key]:>16{fmt}}"
        lines.append(row)

    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
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

    # Top: equity curves
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

    # Metrics annotation
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

    # Bottom: drawdown
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Run Clenow R² Momentum strategy backtest."""
    print("=" * 60)
    print("Clenow R² Momentum Strategy")
    print("=" * 60)

    # 1. Load data
    print("\nLoading data...")
    close_prices, open_prices, tickers = load_data(period="3y")
    print(f"  Universe: {len(tickers)} tickers")
    print(f"  Date range: {close_prices.index[0].date()} to {close_prices.index[-1].date()}")
    print(f"  Lookback: {R2_LOOKBACK}d, KAMA asset: {KAMA_ASSET}d, KAMA SPY: {KAMA_SPY}d, buffer: {KAMA_BUF}")
    print(f"  Gap filter: {GAP_THRESHOLD:.0%}, ATR period: {ATR_PERIOD}d")
    print(f"  Rebalance: every {REBAL_PERIOD_WEEKS} weeks (incremental lazy-hold)")

    # 2. Run backtest
    print("\nRunning R² Momentum backtest...")
    equity, sel_hist, trade_log = run_backtest(
        close_prices, open_prices, tickers,
        initial_capital=INITIAL_CAPITAL, top_n=TOP_N,
        rebal_period_weeks=REBAL_PERIOD_WEEKS,
    )
    m = compute_metrics(equity)
    print(f"  CAGR: {m['cagr']:.1%}  Sharpe: {m['sharpe']:.2f}  MaxDD: {m['max_drawdown']:.1%}")
    print(f"  Total trades: {len(trade_log)}")

    # 3. SPY benchmark
    warmup_days = max(R2_LOOKBACK, KAMA_SPY, KAMA_ASSET) + 10
    sim_dates = close_prices.index[warmup_days:]
    spy_close = close_prices[SPY_TICKER]
    spy_equity = INITIAL_CAPITAL * (
        spy_close.loc[sim_dates] / spy_close.loc[sim_dates].iloc[0]
    )

    # 4. Report
    report = format_report(equity, spy_equity)
    print(f"\n{report}")

    # 5. Save outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "report.txt").write_text(report, encoding="utf-8")

    equity_df = pd.DataFrame({"R2_Momentum": equity, "SPY": spy_equity})
    equity_df.to_csv(OUTPUT_DIR / "equity_curves.csv")

    if trade_log:
        pd.DataFrame(trade_log).to_csv(OUTPUT_DIR / "trade_log.csv", index=False)

    # 6. Plots
    print("\nGenerating plots...")
    plot_equity_and_drawdown(equity, spy_equity, OUTPUT_DIR)
    plot_holdings_over_time(sel_hist, OUTPUT_DIR)
    plot_turnover(trade_log, OUTPUT_DIR)
    plot_asset_frequency(sel_hist, OUTPUT_DIR)

    print(f"\nAll outputs saved to {OUTPUT_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
