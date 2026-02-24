"""Bar-by-bar simulation engine with Long/Short Equity support.

Signals computed on Close(T), execution on Open(T+1).
Market Breathing: SPY vs KAMA(SPY) determines bull/bear regime.
Symmetric KAMA trailing stops for both long and short positions.
Cash accounting model supports negative share positions (shorts).
"""

import numpy as np
import pandas as pd

from src.portfolio_sim.alpha import compute_target_weights
from src.portfolio_sim.config import (
    COMMISSION_RATE,
    REBALANCE_INTERVAL,
    SAFE_HAVEN_TICKER,
    SLIPPAGE_RATE,
    SPY_TICKER,
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
) -> tuple[list[float], list[float], list[float], np.ndarray]:
    """Run a full bar-by-bar portfolio simulation.

    Args:
        sim_prices: Close prices for the simulation period.
        sim_open: Open prices for the simulation period (same index).
        full_prices: Full history of Close prices (for lookback/indicators).
        tickers: master list of tradable tickers (including SHV).
        params: the 3 tunable strategy parameters.
        initial_capital: starting cash.

    Returns:
        equity: daily portfolio values (Close mark-to-market).
        gross_exposures: daily gross exposure ratio (sum of |weights|, excl SHV).
        net_exposures: daily net exposure ratio (sum of signed weights, excl SHV).
        final_weights: last-applied target weights array.
    """
    lookback = params.lookback_period

    equity: list[float] = []
    gross_exposures: list[float] = []
    net_exposures: list[float] = []

    # Cash accounting model
    cash = initial_capital
    shares: dict[str, float] = {t: 0.0 for t in tickers}
    current_weights = np.zeros(len(tickers))

    # ------------------------------------------------------------------
    # 0. Pre-compute KAMA for all tickers on full history
    # ------------------------------------------------------------------
    kama_cache: dict[str, pd.Series] = {}
    for t in list(set(tickers + [SPY_TICKER])):
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

    # Determine initial regime from SPY
    spy_kama_init = kama_cache.get(SPY_TICKER, pd.Series(dtype=float)).get(prev_date, np.nan)
    spy_price_init = (
        full_prices[SPY_TICKER].iloc[idx_start - 1]
        if SPY_TICKER in full_prices.columns
        else np.nan
    )
    if np.isnan(spy_kama_init) or np.isnan(spy_price_init):
        initial_is_bull = True  # Default to bull during warm-up
    else:
        initial_is_bull = bool(spy_price_init > spy_kama_init)

    pending_weights = compute_target_weights(
        past_prices_init, tickers, params, kama_init, is_bull=initial_is_bull
    )

    # Park remainder in SHV
    net_invested = pending_weights.sum()
    if SAFE_HAVEN_TICKER in tickers:
        safe_idx = tickers.index(SAFE_HAVEN_TICKER)
        pending_weights[safe_idx] = 1.0 - net_invested

    prev_is_bull: bool | None = initial_is_bull

    # ------------------------------------------------------------------
    # 2. Day-by-day loop
    # ------------------------------------------------------------------
    for i, (date, daily_prices) in enumerate(sim_prices.iterrows()):
        open_prices = sim_open.loc[date].fillna(daily_prices)

        # --- Execute pending orders on Open ---
        if pending_weights is not None:
            # Current equity at open prices
            equity_at_open = cash + sum(
                shares[t] * open_prices[t] for t in tickers
            )
            if equity_at_open <= 0:
                equity_at_open = max(cash, 1.0)

            turnover = 0.0
            new_shares: dict[str, float] = {}

            for j, t in enumerate(tickers):
                target_val = equity_at_open * pending_weights[j]
                current_val = shares[t] * open_prices[t]
                turnover += abs(target_val - current_val)

                if open_prices[t] > 0:
                    new_shares[t] = target_val / open_prices[t]
                else:
                    new_shares[t] = 0.0

            cost = turnover * (COMMISSION_RATE + SLIPPAGE_RATE)

            # Update cash: remainder after deploying weights + deduct costs
            cash = equity_at_open * (1.0 - pending_weights.sum()) - cost
            shares = new_shares
            current_weights = pending_weights
            pending_weights = None

        # --- Mark-to-market on Close ---
        equity_value = cash + sum(
            shares[t] * daily_prices[t] for t in tickers
        )

        if np.isnan(equity_value) or equity_value <= 0.001 * initial_capital:
            equity_value = 0.0
            remaining = len(sim_prices) - len(equity)
            equity.extend([0.0] * remaining)
            gross_exposures.extend([0.0] * remaining)
            net_exposures.extend([0.0] * remaining)
            break

        equity.append(equity_value)

        # Exposure tracking (excl SHV)
        denom = equity_value if abs(equity_value) > 1e-6 else 1.0
        gross_exp = sum(
            abs(shares[t] * daily_prices[t])
            for t in tickers
            if t != SAFE_HAVEN_TICKER
        ) / denom
        net_exp = sum(
            shares[t] * daily_prices[t]
            for t in tickers
            if t != SAFE_HAVEN_TICKER
        ) / denom
        gross_exposures.append(gross_exp)
        net_exposures.append(net_exp)

        # ------------------------------------------------------------------
        # 3. Compute signals on Close(T) for execution on Open(T+1)
        # ------------------------------------------------------------------

        # Market Breathing: SPY vs KAMA(SPY)
        spy_kama = kama_cache.get(SPY_TICKER, pd.Series(dtype=float)).get(date, np.nan)
        spy_close = daily_prices.get(SPY_TICKER, np.nan) if SPY_TICKER in daily_prices.index else np.nan

        if np.isnan(spy_kama) or np.isnan(spy_close):
            is_bull = True  # Default to bull during warm-up
        else:
            is_bull = bool(spy_close > spy_kama)

        # Regime change detection: force rebalance on transition
        regime_changed = (prev_is_bull is not None) and (is_bull != prev_is_bull)
        prev_is_bull = is_bull

        # Symmetric KAMA trailing stops
        should_rebalance = regime_changed or (i > 0 and i % REBALANCE_INTERVAL == 0)

        stop_tickers: list[str] = []
        for t in tickers:
            if t == SAFE_HAVEN_TICKER or shares[t] == 0.0:
                continue
            t_kama = kama_cache[t].get(date, np.nan) if t in kama_cache else np.nan
            if np.isnan(t_kama):
                continue

            if shares[t] > 0 and daily_prices[t] < t_kama:
                # Long position stopped: price dropped below KAMA
                stop_tickers.append(t)
            elif shares[t] < 0 and daily_prices[t] > t_kama:
                # Short position stopped: price rose above KAMA
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
                past_prices, tickers, params, kama_current, is_bull=is_bull
            )

            # Park remainder in SHV
            net_invested = new_weights.sum()
            if SAFE_HAVEN_TICKER in tickers:
                safe_idx = tickers.index(SAFE_HAVEN_TICKER)
                new_weights[safe_idx] = 1.0 - net_invested

            pending_weights = new_weights

        elif stop_tickers:
            # Partial rebalance: exit stopped positions to SHV
            new_weights = np.zeros(len(tickers))
            freed_weight = 0.0

            for j, t in enumerate(tickers):
                w = (
                    (shares[t] * daily_prices[t]) / equity_value
                    if equity_value > 0
                    else 0.0
                )
                if t in stop_tickers:
                    freed_weight += abs(w)
                else:
                    new_weights[j] = w

            if SAFE_HAVEN_TICKER in tickers and freed_weight > 0:
                safe_idx = tickers.index(SAFE_HAVEN_TICKER)
                new_weights[safe_idx] += freed_weight

            total = new_weights.sum()
            if total > 0:
                new_weights /= total

            pending_weights = new_weights

    return equity, gross_exposures, net_exposures, current_weights
