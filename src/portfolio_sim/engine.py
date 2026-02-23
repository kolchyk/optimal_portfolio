"""Bar-by-bar simulation engine.

Signals computed on Close(T), execution on Open(T+1).
Includes Market Breadth filter (Step 1), KAMA trailing stops, and SHV parking.
"""

import numpy as np
import pandas as pd

from src.portfolio_sim.alpha import compute_target_weights
from src.portfolio_sim.config import (
    BREADTH_THRESHOLD,
    COMMISSION_RATE,
    REBALANCE_INTERVAL,
    SAFE_HAVEN_TICKER,
    SLIPPAGE_RATE,
    StrategyParams,
)
from src.portfolio_sim.indicators import compute_kama_series


def run_simulation(
    sim_prices: pd.DataFrame,
    sim_open: pd.DataFrame,
    full_prices: pd.DataFrame,
    tickers: list[str],
    params: StrategyParams,
    initial_capital: float,
) -> tuple[list[float], list[float], np.ndarray]:
    """Run a full bar-by-bar portfolio simulation.

    Args:
        sim_prices: Close prices for the simulation period.
        sim_open: Open prices for the simulation period (same index).
        full_prices: Full history of Close prices (for lookback/indicators).
        tickers: master list of tradable tickers (including SHV).
        params: the 4 tunable strategy parameters.
        initial_capital: starting cash.

    Returns:
        equity: daily portfolio values (Close mark-to-market).
        long_exposures: daily long exposure ratio (excl SHV).
        final_weights: last-applied target weights array.
    """
    lookback = params.lookback_period

    equity: list[float] = []
    long_exposures: list[float] = []
    val_alpha = initial_capital

    # ------------------------------------------------------------------
    # 0. Pre-compute KAMA for all tickers on full history
    # ------------------------------------------------------------------
    kama_cache: dict[str, pd.Series] = {}
    for t in tickers + ["SPY"]:
        if t in full_prices.columns:
            kama_cache[t] = compute_kama_series(
                full_prices[t].dropna(), period=params.kama_period
            )

    # ------------------------------------------------------------------
    # 1. Initial signal (on Close of the day before sim start)
    # ------------------------------------------------------------------
    idx_start = full_prices.index.get_loc(sim_prices.index[0])
    past_prices_init = full_prices.iloc[idx_start - lookback : idx_start][tickers]

    prev_date = full_prices.index[idx_start - 1]
    kama_init = {
        t: kama_cache[t].get(prev_date, 0)
        for t in tickers
        if t in kama_cache and not np.isnan(kama_cache[t].get(prev_date, np.nan))
    }

    pending_weights = compute_target_weights(
        past_prices_init, tickers, params, kama_init
    )

    # Park remainder in SHV
    total_w = pending_weights.sum()
    if total_w < 1.0 and SAFE_HAVEN_TICKER in tickers:
        safe_idx = tickers.index(SAFE_HAVEN_TICKER)
        pending_weights[safe_idx] += 1.0 - total_w

    shares: dict[str, float] = {t: 0.0 for t in tickers}
    current_weights = np.zeros(len(tickers))

    # ------------------------------------------------------------------
    # 2. Day-by-day loop
    # ------------------------------------------------------------------
    for i, (date, daily_prices) in enumerate(sim_prices.iterrows()):
        open_prices = sim_open.loc[date].fillna(daily_prices)

        # --- Execute pending orders on Open ---
        if pending_weights is not None:
            val_at_open = sum(shares[t] * open_prices[t] for t in tickers)
            if val_at_open <= 0:
                val_at_open = val_alpha

            turnover = 0.0
            new_shares: dict[str, float] = {}
            tolerance = 0.02

            for j, t in enumerate(tickers):
                target_w = pending_weights[j]
                current_w = (
                    (shares[t] * open_prices[t]) / val_at_open
                    if val_at_open > 0
                    else 0.0
                )

                if abs(target_w - current_w) > tolerance or target_w == 0:
                    target_val = val_at_open * target_w
                    current_val = shares[t] * open_prices[t]
                    turnover += abs(target_val - current_val)
                    new_shares[t] = target_val / open_prices[t] if open_prices[t] > 0 else 0.0
                else:
                    new_shares[t] = shares[t]

            val_alpha = val_at_open - turnover * (COMMISSION_RATE + SLIPPAGE_RATE)
            val_alpha = max(0.0, val_alpha)

            multiplier = (val_alpha / val_at_open) if val_at_open > 1e-6 else 0.0
            shares = {t: new_shares[t] * multiplier for t in tickers}
            current_weights = pending_weights
            pending_weights = None

        # --- Mark-to-market on Close ---
        val_alpha = sum(shares[t] * daily_prices[t] for t in tickers)

        if np.isnan(val_alpha) or val_alpha <= 0.001:
            val_alpha = 0.0
            remaining = len(sim_prices) - len(equity)
            equity.extend([0.0] * remaining)
            long_exposures.extend([0.0] * remaining)
            break

        equity.append(val_alpha)

        # Long exposure (excl SHV)
        denom = val_alpha if abs(val_alpha) > 1e-6 else 1.0
        gross_long = (
            sum(
                shares[t] * daily_prices[t]
                for t in tickers
                if t != SAFE_HAVEN_TICKER
            )
            / denom
        )
        long_exposures.append(gross_long)

        # ------------------------------------------------------------------
        # 3. Compute signals on Close(T) for execution on Open(T+1)
        # ------------------------------------------------------------------

        # Step 1: Market Breadth filter
        active_uptrends = 0
        total_valid = 0
        for t in tickers:
            if t == SAFE_HAVEN_TICKER:
                continue
            t_kama = kama_cache.get(t, pd.Series(dtype=float)).get(date, np.nan)
            if not np.isnan(t_kama):
                total_valid += 1
                if daily_prices[t] > t_kama:
                    active_uptrends += 1

        breadth = active_uptrends / max(1, total_valid)
        is_bull = breadth >= BREADTH_THRESHOLD

        if not is_bull:
            # All to SHV
            new_weights = np.zeros(len(tickers))
            if SAFE_HAVEN_TICKER in tickers:
                safe_idx = tickers.index(SAFE_HAVEN_TICKER)
                new_weights[safe_idx] = 1.0
            pending_weights = new_weights
            continue

        # Individual KAMA trailing stops
        should_rebalance = i > 0 and i % REBALANCE_INTERVAL == 0

        stop_tickers: list[str] = []
        for t in tickers:
            if t == SAFE_HAVEN_TICKER or shares[t] <= 0:
                continue
            t_kama = kama_cache.get(t, pd.Series(dtype=float)).get(date, np.nan)
            if not np.isnan(t_kama) and daily_prices[t] < t_kama:
                stop_tickers.append(t)

        if should_rebalance:
            # Full rebalance
            idx_in_full = full_prices.index.get_loc(date)
            past_prices = full_prices.iloc[
                idx_in_full - lookback + 1 : idx_in_full + 1
            ][tickers]

            kama_current = {
                t: kama_cache[t].get(date, 0)
                for t in tickers
                if t in kama_cache
                and not np.isnan(kama_cache[t].get(date, np.nan))
            }

            new_weights = compute_target_weights(
                past_prices, tickers, params, kama_current
            )

            total_invested = new_weights.sum()
            if total_invested < 1.0 and SAFE_HAVEN_TICKER in tickers:
                safe_idx = tickers.index(SAFE_HAVEN_TICKER)
                new_weights[safe_idx] += 1.0 - total_invested

            pending_weights = new_weights

        elif stop_tickers:
            # Partial rebalance: exit stopped positions to SHV
            new_weights = np.zeros(len(tickers))
            freed_weight = 0.0

            for j, t in enumerate(tickers):
                w = (
                    (shares[t] * daily_prices[t]) / val_alpha
                    if val_alpha > 0
                    else 0.0
                )
                if t in stop_tickers:
                    freed_weight += w
                else:
                    new_weights[j] = w

            if SAFE_HAVEN_TICKER in tickers and freed_weight > 0:
                safe_idx = tickers.index(SAFE_HAVEN_TICKER)
                new_weights[safe_idx] += freed_weight

            total = new_weights.sum()
            if total > 0:
                new_weights /= total

            pending_weights = new_weights

    return equity, long_exposures, current_weights
