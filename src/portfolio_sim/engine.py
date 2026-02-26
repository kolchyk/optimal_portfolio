"""Bar-by-bar simulation engine — Long/Cash only.

Signals computed on Close(T), execution on Open(T+1).

Key design decisions that prevent capital destruction:
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
        SimulationResult with equity curves, holdings history, and trades.
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
    # FIX: Remove the hard 150-row (warmup) restriction.
    # If the provided data is too short, we start from the very first bar.
    # Indicators will naturally stay empty until enough bars are accumulated.
    warmup = p.warmup
    if len(close_prices) <= warmup:
        warmup = 0

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

    # Tracking for dashboard
    cash_values: list[float] = []
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
        holdings_rows.append({t: shares.get(t, 0.0) for t in tickers})

        if equity_value <= 0:
            remaining = len(sim_dates) - i - 1
            equity_values.extend([0.0] * remaining)
            cash_values.extend([0.0] * remaining)
            empty_row = {t: 0.0 for t in tickers}
            holdings_rows.extend([empty_row] * remaining)
            break

        # --- Compute signals on Close(T) for Open(T+1) ---

        # Sell ONLY on individual KAMA stop-loss.
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

    return SimulationResult(
        equity=equity_series,
        spy_equity=spy_series,
        holdings_history=holdings_df,
        cash_history=cash_series,
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
      - trades[t] == 0.0: sell position t.
      - trades[t] == 1.0: buy position t.

    When *weights* is provided (risk parity mode), each buy's allocation is
    equity_at_open * weights[t]. Otherwise strict 1/top_n equal weight.
    *max_weight* caps any single position at equity_at_open * max_weight.
    """
    total_cost = 0.0

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
