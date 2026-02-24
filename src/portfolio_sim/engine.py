"""Bar-by-bar simulation engine — Long/Cash only.

Signals computed on Close(T), execution on Open(T+1).
Market Breathing: SPY vs KAMA(SPY) determines risk-on/risk-off.
KAMA stop-loss with buffer for individual positions.
Lazy execution: don't sell good positions just to rebalance.
"""

import numpy as np
import pandas as pd

from src.portfolio_sim.alpha import compute_target_weights
from src.portfolio_sim.config import (
    COMMISSION_RATE,
    KAMA_BUFFER,
    KAMA_PERIOD,
    LOOKBACK_PERIOD,
    SLIPPAGE_RATE,
    SPY_TICKER,
)
from src.portfolio_sim.indicators import compute_kama_series

COST_RATE = COMMISSION_RATE + SLIPPAGE_RATE


def run_simulation(
    close_prices: pd.DataFrame,
    open_prices: pd.DataFrame,
    tickers: list[str],
    initial_capital: float,
) -> tuple[pd.Series, pd.Series]:
    """Run a full bar-by-bar portfolio simulation.

    Args:
        close_prices: Full history of Close prices (DatetimeIndex rows,
                      ticker columns). Must include SPY.
        open_prices: Full history of Open prices (same shape/index).
        tickers: list of tradable ticker symbols (excludes SPY).
        initial_capital: starting cash.

    Returns:
        equity: pd.Series of daily portfolio value (Close mark-to-market).
        spy_equity: pd.Series of SPY buy-and-hold equity for benchmark.
    """
    # ------------------------------------------------------------------
    # 0. Pre-compute KAMA for all tickers + SPY
    # ------------------------------------------------------------------
    kama_cache: dict[str, pd.Series] = {}
    for t in list(set(tickers + [SPY_TICKER])):
        if t in close_prices.columns:
            kama_cache[t] = compute_kama_series(
                close_prices[t].dropna(), period=KAMA_PERIOD
            )

    # ------------------------------------------------------------------
    # 1. Determine simulation start (need warm-up for KAMA + lookback)
    # ------------------------------------------------------------------
    warmup = max(LOOKBACK_PERIOD, KAMA_PERIOD) + 10
    if len(close_prices) <= warmup:
        raise ValueError(f"Need at least {warmup} rows, got {len(close_prices)}")

    sim_dates = close_prices.index[warmup:]

    # SPY buy-and-hold benchmark
    spy_prices = close_prices[SPY_TICKER].loc[sim_dates]
    spy_equity = initial_capital * (spy_prices / spy_prices.iloc[0])

    # ------------------------------------------------------------------
    # 2. State
    # ------------------------------------------------------------------
    cash = initial_capital
    shares: dict[str, float] = {}  # ticker -> num shares (only held positions)
    equity_values: list[float] = []

    # Pending trades: dict of ticker -> target_weight, or None.
    # Special: empty dict {} means "sell everything" (bear regime).
    pending_trades: dict[str, float] | None = None

    # ------------------------------------------------------------------
    # 3. Day-by-day loop
    # ------------------------------------------------------------------
    for i, date in enumerate(sim_dates):
        daily_close = close_prices.loc[date]
        daily_open = open_prices.loc[date]

        # --- Execute pending trades on Open ---
        if pending_trades is not None:
            equity_at_open = cash + sum(
                shares[t] * daily_open.get(t, 0.0) for t in shares
            )
            if equity_at_open <= 0:
                equity_at_open = max(cash, 1.0)

            cash = _execute_trades(shares, pending_trades, equity_at_open, daily_open)
            pending_trades = None

        # --- Mark-to-market on Close ---
        equity_value = cash + sum(
            shares[t] * daily_close.get(t, 0.0) for t in shares
        )
        equity_values.append(equity_value)

        if equity_value <= 0:
            equity_values.extend([0.0] * (len(sim_dates) - i - 1))
            break

        # --- Compute signals on Close(T) for Open(T+1) ---

        # Market Breathing: SPY vs KAMA(SPY)
        spy_close = daily_close.get(SPY_TICKER, np.nan)
        spy_kama_s = kama_cache.get(SPY_TICKER, pd.Series(dtype=float))
        spy_kama = spy_kama_s.get(date, np.nan) if date in spy_kama_s.index else np.nan
        if np.isnan(spy_close) or np.isnan(spy_kama):
            is_bull = True
        else:
            is_bull = bool(spy_close > spy_kama)

        # Bear regime: sell everything
        if not is_bull and shares:
            pending_trades = {}
            continue

        # KAMA stop-loss: identify positions to sell
        sells: dict[str, float] = {}
        for t in list(shares.keys()):
            t_kama_s = kama_cache.get(t, pd.Series(dtype=float))
            t_kama = t_kama_s.get(date, np.nan) if date in t_kama_s.index else np.nan
            if np.isnan(t_kama):
                continue
            if daily_close.get(t, 0.0) < t_kama * (1 - KAMA_BUFFER):
                sells[t] = 0.0  # mark for selling

        # Get target portfolio from alpha
        idx_in_full = close_prices.index.get_loc(date)
        start = max(0, idx_in_full - LOOKBACK_PERIOD + 1)
        past_prices = close_prices.iloc[start : idx_in_full + 1][tickers]

        kama_current: dict[str, float] = {}
        for t in tickers:
            if t in kama_cache:
                t_kama_s = kama_cache[t]
                val = t_kama_s.get(date, np.nan) if date in t_kama_s.index else np.nan
                if not np.isnan(val):
                    kama_current[t] = val

        target_weights = compute_target_weights(
            past_prices, tickers, kama_current, is_bull=is_bull
        )

        # --- Lazy execution logic ---
        # Sell: stopped-out positions or positions no longer in target
        # Buy: new target positions not already held
        # Keep: existing positions still in target (don't touch)

        new_trades: dict[str, float] = {}

        # Mark stopped positions for sell
        for t in sells:
            if t in shares:
                new_trades[t] = 0.0

        # Mark positions not in target for sell (only if they're also not stopped —
        # stopped ones are already marked above)
        for t in list(shares.keys()):
            if t not in target_weights and t not in new_trades:
                new_trades[t] = 0.0

        # Buy new positions that are in target but not already held
        for t in target_weights:
            if t not in shares:
                new_trades[t] = target_weights[t]

        if new_trades:
            pending_trades = new_trades

    equity_series = pd.Series(
        equity_values, index=sim_dates[: len(equity_values)]
    )
    spy_series = spy_equity.iloc[: len(equity_values)]

    return equity_series, spy_series


def _execute_trades(
    shares: dict[str, float],
    trades: dict[str, float],
    equity_at_open: float,
    open_prices: pd.Series,
) -> float:
    """Execute trades: sell positions with weight=0, buy new ones.

    Mutates `shares` in place. Returns remaining cash.

    If trades is empty dict: sell everything (bear regime / all cash).
    Otherwise: sell positions with target=0.0, buy positions with target>0.
    """
    total_cost = 0.0

    # Sell everything (bear regime)
    if not trades:
        cash = 0.0
        for t in list(shares.keys()):
            price = open_prices.get(t, 0.0)
            if price > 0 and shares[t] > 0:
                trade_value = shares[t] * price
                total_cost += trade_value * COST_RATE
                cash += trade_value
            del shares[t]
        return equity_at_open - sum(s * open_prices.get(t, 0.0) for t, s in shares.items()) - total_cost + cash
        # Simplified: everything is sold, cash = equity - costs

    # Partial trades: sell some, buy some
    freed_cash = 0.0

    # First: execute sells
    for t, w in list(trades.items()):
        if w == 0.0 and t in shares:
            price = open_prices.get(t, 0.0)
            if price > 0:
                trade_value = shares[t] * price
                freed_cash += trade_value
                total_cost += trade_value * COST_RATE
            del shares[t]

    # Compute current cash position
    held_value = sum(shares[t] * open_prices.get(t, 0.0) for t in shares)
    cash = equity_at_open - held_value

    # Available for new buys = freed cash + existing cash - costs so far
    available = cash - total_cost

    # Execute buys: allocate available cash equally among new positions
    buys = {t: w for t, w in trades.items() if w > 0 and t not in shares}
    if buys and available > 0:
        per_position = available / len(buys)
        for t in buys:
            price = open_prices.get(t, 0.0)
            if price > 0:
                num_shares = per_position / price
                shares[t] = num_shares
                total_cost += per_position * COST_RATE

    # Final cash
    held_value = sum(shares[t] * open_prices.get(t, 0.0) for t in shares)
    return equity_at_open - held_value - total_cost
