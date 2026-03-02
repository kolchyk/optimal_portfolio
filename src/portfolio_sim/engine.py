"""Hybrid bar-by-bar simulation engine.

Combines the best elements of R2 (Clenow R² Momentum) and vol-targeting:
  - R² Momentum scoring: annualized OLS slope × R² for asset ranking
  - KAMA trend filter + per-class sector limits for entry
  - ATR risk parity position sizing (Clenow-style)
  - Hybrid rebalancing: periodic rotation + daily KAMA-break/gap exits
  - Vol-targeting overlay: scales positions by target_vol / realised_vol
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from numba import njit
from tqdm import tqdm

from src.portfolio_sim.config import (
    ASSET_CLASS_MAP,
    COMMISSION_RATE,
    MARGIN_SPREAD,
    RISK_FREE_RATE,
    SLIPPAGE_RATE,
    SPY_TICKER,
)
from src.portfolio_sim.indicators import compute_kama_series
from src.portfolio_sim.models import SimulationResult
from src.portfolio_sim.params import StrategyParams

COST_RATE = COMMISSION_RATE + SLIPPAGE_RATE
DAILY_RF = (1 + RISK_FREE_RATE) ** (1 / 252) - 1


# ---------------------------------------------------------------------------
# Vectorized rolling R² Momentum scoring
# ---------------------------------------------------------------------------
@njit(cache=True)
def _compute_rolling_r2_scores(log_prices: np.ndarray, lookback: int) -> np.ndarray:
    """Compute rolling R² Momentum scores for a single ticker.

    For each window of *lookback* bars, fits OLS on log-prices and returns
    score = annualized_return × R².

    Returns array of same length as log_prices, NaN where insufficient data.
    """
    n = len(log_prices)
    scores = np.full(n, np.nan)

    x = np.arange(lookback, dtype=np.float64)
    x_mean = x.mean()
    ss_xx = ((x - x_mean) ** 2).sum()

    for t in range(lookback - 1, n):
        window = log_prices[t - lookback + 1: t + 1]
        if np.any(np.isnan(window)) or np.any(np.isinf(window)):
            continue

        y_mean = window.mean()
        ss_yy = ((window - y_mean) ** 2).sum()
        if ss_yy < 1e-10:
            continue

        ss_xy = ((x - x_mean) * (window - y_mean)).sum()
        slope = ss_xy / ss_xx

        r_squared = (ss_xy ** 2) / (ss_xx * ss_yy)
        r_squared = max(0.0, min(1.0, r_squared))

        annualized_return = float(np.exp(slope * 252) - 1.0)
        scores[t] = annualized_return * r_squared

    return scores


def _precompute_r2_score_df(
    close_prices: pd.DataFrame,
    ticker_cols: list[str],
    r2_window: int,
) -> pd.DataFrame:
    """Pre-compute R² momentum scores for a single lookback window.

    For each ticker, computes annualized_return × R² using OLS on log-prices.
    """
    score_df = pd.DataFrame(np.nan, index=close_prices.index, columns=ticker_cols)
    for t in ticker_cols:
        ser = close_prices[t].dropna()
        if len(ser) < r2_window:
            continue
        log_p = np.log(ser.values)
        scores = _compute_rolling_r2_scores(log_p, r2_window)
        score_df.loc[ser.index, t] = scores
    return score_df


# ---------------------------------------------------------------------------
# ATR risk parity sizing
# ---------------------------------------------------------------------------
def _compute_atr_weights(
    tickers_to_buy: list[str],
    atr_row: pd.Series,
    price_row: pd.Series,
    risk_factor: float,
) -> dict[str, float]:
    """Compute ATR-based inverse risk parity weights (Clenow-style).

    weight_i = (risk_factor / ATR%_i) / sum(risk_factor / ATR%_j)
    where ATR% = ATR / price.
    """
    if not tickers_to_buy:
        return {}

    atr_inv: dict[str, float] = {}
    for t in tickers_to_buy:
        atr = atr_row.get(t, np.nan)
        price = price_row.get(t, np.nan)
        if np.isnan(atr) or np.isnan(price) or atr < 1e-10 or price < 1e-10:
            atr_inv[t] = 1.0
        else:
            atr_pct = atr / price
            atr_inv[t] = risk_factor / atr_pct

    total = sum(atr_inv.values())
    if total < 1e-10:
        w = 1.0 / len(tickers_to_buy)
        return {t: w for t in tickers_to_buy}
    return {t: v / total for t, v in atr_inv.items()}


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------
def run_simulation(
    close_prices: pd.DataFrame,
    open_prices: pd.DataFrame,
    tickers: list[str],
    initial_capital: float,
    params: StrategyParams | None = None,
    kama_cache: dict[str, pd.Series] | None = None,
    show_progress: bool = False,
    high_prices: pd.DataFrame | None = None,
    low_prices: pd.DataFrame | None = None,
) -> SimulationResult:
    """Run hybrid bar-by-bar simulation.

    Combines R² Momentum scoring, ATR risk parity sizing, KAMA filters,
    gap-based exits, and vol-targeting overlay.
    """
    p = params or StrategyParams()

    # ------------------------------------------------------------------
    # 0. Pre-compute KAMA for all tickers + SPY
    # ------------------------------------------------------------------
    if kama_cache is None:
        kama_cache = {}
        all_tickers = list(set(tickers) | {SPY_TICKER})
        for t in all_tickers:
            if t in close_prices.columns:
                kama_cache[t] = compute_kama_series(
                    close_prices[t].dropna(), period=p.kama_asset_period,
                )

    # ------------------------------------------------------------------
    # 0b. Pre-compute vectorised matrices
    # ------------------------------------------------------------------
    _ticker_cols = [t for t in tickers if t in close_prices.columns]
    _prices = close_prices[_ticker_cols]

    # R² Momentum scores (single lookback window)
    r2_score_df = _precompute_r2_score_df(_prices, _ticker_cols, p.r2_window)

    # True ATR: max(H-L, |H-C_prev|, |L-C_prev|), rolling mean
    if high_prices is not None and low_prices is not None:
        _high = high_prices[_ticker_cols]
        _low = low_prices[_ticker_cols]
        _prev_close = _prices.shift(1)
        tr1 = _high - _low
        tr2 = (_high - _prev_close).abs()
        tr3 = (_low - _prev_close).abs()
        true_range = np.maximum(np.maximum(tr1, tr2), tr3)
        atr_df = true_range.rolling(p.atr_period, min_periods=1).mean()
    else:
        # Fallback: close-to-close when high/low not available
        atr_df = _prices.diff().abs().rolling(p.atr_period, min_periods=1).mean()

    # Returns for correlation filter
    returns_df = _prices.pct_change()

    # KAMA as DataFrame for fast row access
    kama_df = pd.DataFrame(kama_cache) if kama_cache else pd.DataFrame()

    # ------------------------------------------------------------------
    # 1. Determine simulation start
    # ------------------------------------------------------------------
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
    cash_values: list[float] = []
    holdings_rows: list[dict[str, float]] = []
    trade_log: list[dict] = []

    pending_trades: dict[str, float] | None = None
    pending_weights: dict[str, float] | None = None

    # Vol-targeting state
    portfolio_returns: list[float] = []
    current_scale: float = 1.0

    # Rebalance schedule
    rebal_interval = p.rebal_days
    rebalance_dates = set(sim_dates[::rebal_interval])
    rebalance_dates.add(sim_dates[0])

    # ------------------------------------------------------------------
    # 3. Day-by-day loop
    # ------------------------------------------------------------------
    date_iter = (
        tqdm(sim_dates, desc="Simulating", unit="day")
        if show_progress
        else sim_dates
    )
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
            shares_before_counts = {t: shares[t] for t in shares}
            cash = _execute_trades(
                shares, pending_trades, equity_at_open, daily_open,
                weights=pending_weights, scale=current_scale,
            )

            # Log trades
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
            for t in shares_before & set(shares.keys()):
                old_shares = shares_before_counts.get(t, 0.0)
                if shares[t] < old_shares - 1e-10:
                    trade_log.append({
                        "date": date, "ticker": t, "action": "trim",
                        "shares": shares[t], "price": daily_open.get(t, 0.0),
                    })
            pending_trades = None
            pending_weights = None

        # --- Vol-targeting trim on Open ---
        if shares:
            equity_at_open = cash + sum(
                shares[t] * daily_open.get(t, 0.0) for t in shares
            )
            if equity_at_open > 0:
                held_value = sum(
                    shares[t] * daily_open.get(t, 0.0) for t in shares
                )
                current_fraction = held_value / equity_at_open
                target_fraction = current_scale

                if current_fraction > target_fraction + 0.05:
                    trim_ratio = target_fraction / current_fraction
                    trim_cost = 0.0
                    for t in list(shares.keys()):
                        price = daily_open.get(t, 0.0)
                        if price <= 0:
                            continue
                        old_shares = shares[t]
                        new_shares = old_shares * trim_ratio
                        sold_shares = old_shares - new_shares
                        if sold_shares > 1e-10:
                            sell_value = sold_shares * price
                            trim_cost += sell_value * COST_RATE
                            shares[t] = new_shares
                            trade_log.append({
                                "date": date, "ticker": t, "action": "trim",
                                "shares": new_shares, "price": price,
                            })
                    held_value_new = sum(
                        shares[t] * daily_open.get(t, 0.0) for t in shares
                    )
                    cash = equity_at_open - held_value_new - trim_cost

        # --- Cash earns risk-free rate / margin cost ---
        if cash > 0:
            cash *= 1 + DAILY_RF
        elif cash < 0:
            margin_rate = DAILY_RF + MARGIN_SPREAD / 252
            cash *= 1 + margin_rate

        # --- Mark-to-market on Close ---
        equity_value = cash + sum(
            shares[t] * daily_close.get(t, 0.0) for t in shares
        )
        equity_values.append(equity_value)
        cash_values.append(cash)
        holdings_rows.append({t: shares.get(t, 0.0) for t in tickers})

        if equity_value <= 0:
            remaining = len(sim_dates) - i - 1
            equity_values.extend([0.0] * remaining)
            cash_values.extend([0.0] * remaining)
            empty_row = {t: 0.0 for t in tickers}
            holdings_rows.extend([empty_row] * remaining)
            break

        # --- Update portfolio return tracker ---
        if i > 0:
            prev_eq = equity_values[-2]
            if prev_eq > 0:
                portfolio_returns.append(equity_value / prev_eq - 1.0)

        # --- Compute vol-targeting scale for next day ---
        if len(portfolio_returns) >= p.portfolio_vol_lookback:
            recent_rets = portfolio_returns[-p.portfolio_vol_lookback:]
            realized_vol = np.std(recent_rets) * np.sqrt(252)
            if realized_vol > 1e-8:
                current_scale = p.target_vol / realized_vol
                current_scale = max(0.1, min(current_scale, p.max_leverage))
            else:
                current_scale = p.max_leverage

        # --- DAILY exit checks (fast response) ---
        sells: dict[str, float] = {}
        kama_row = (
            kama_df.loc[date]
            if (not kama_df.empty and date in kama_df.index)
            else pd.Series(dtype=float)
        )

        for t in list(shares.keys()):
            # Exit 1: KAMA trend break
            t_kama = kama_row.get(t, np.nan)
            if not np.isnan(t_kama):
                if daily_close.get(t, 0.0) < t_kama * (1 - p.kama_buffer):
                    sells[t] = 0.0
                    continue

        # --- PERIODIC rotation (every N weeks) ---
        is_rebalance_day = date in rebalance_dates
        candidates: list[str] = []

        if is_rebalance_day:
            # Get R² scores for today
            r2_row = (
                r2_score_df.loc[date]
                if date in r2_score_df.index
                else pd.Series(dtype=float)
            )

            scan_tickers = _ticker_cols

            # Apply KAMA trend filter + R² score filter
            scored: dict[str, float] = {}
            for t in scan_tickers:
                # KAMA filter: Close > KAMA * (1 + buffer)
                t_kama = kama_row.get(t, np.nan)
                if not np.isnan(t_kama):
                    if daily_close.get(t, 0.0) <= t_kama * (1 + p.kama_buffer):
                        continue

                score = r2_row.get(t, np.nan)
                if not np.isnan(score) and score > 0:
                    scored[t] = score

            # Rank by R² score (descending)
            all_ranked = sorted(scored, key=scored.get, reverse=True)

            # Extended list for rotation stop
            extended_top = set(all_ranked[: p.top_n * 2])

            # Rotation stop: held asset fell out of extended ranking
            for t in list(shares.keys()):
                if t not in sells and t not in extended_top:
                    sells[t] = 0.0

            # Per-class position count (held after sells)
            held_class_count = Counter(
                ASSET_CLASS_MAP.get(t, "US Equity")
                for t in shares if t not in sells
            )

            # Select candidates with per-class limits
            candidates = []
            for t in all_ranked:
                if t in shares and t not in sells:
                    continue
                t_class = ASSET_CLASS_MAP.get(t, "US Equity")
                if held_class_count[t_class] >= p.max_per_class:
                    continue
                candidates.append(t)
                held_class_count[t_class] += 1
                if len(candidates) >= p.top_n:
                    break

        # --- Build trade instructions ---
        new_trades: dict[str, float] = {}

        for t in sells:
            new_trades[t] = 0.0

        if is_rebalance_day:
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

            buys_list = [t for t, v in new_trades.items() if v == 1.0]
            if buys_list:
                # ATR risk parity weights
                atr_row = (
                    atr_df.loc[date] if date in atr_df.index
                    else pd.Series(dtype=float)
                )
                price_row = daily_close
                pending_weights = _compute_atr_weights(
                    buys_list, atr_row, price_row, p.risk_factor,
                )

    # ------------------------------------------------------------------
    # 4. Build output
    # ------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Trade execution
# ---------------------------------------------------------------------------
def _execute_trades(
    shares: dict[str, float],
    trades: dict[str, float],
    equity_at_open: float,
    open_prices: pd.Series,
    weights: dict[str, float] | None = None,
    scale: float = 1.0,
) -> float:
    """Execute trades with vol-targeting scale.  Mutates ``shares``.

    Returns remaining cash after trades.
    """
    total_cost = 0.0

    # Sells first
    for t, action in list(trades.items()):
        if action == 0.0 and t in shares:
            price = open_prices.get(t, 0.0)
            if price > 0:
                trade_value = shares[t] * price
                total_cost += trade_value * COST_RATE
            del shares[t]

    held_value = sum(shares[t] * open_prices.get(t, 0.0) for t in shares)
    available = equity_at_open - held_value - total_cost

    # Scale budget for new buys by vol-targeting (scale already bounded by max_leverage)
    target_invested = equity_at_open * scale
    budget_for_buys = max(0.0, target_invested - held_value - total_cost)

    buys = [t for t, action in trades.items() if action == 1.0 and t not in shares]
    if buys and budget_for_buys > 0:
        remaining_budget = budget_for_buys
        for t in buys:
            price = open_prices.get(t, 0.0)
            if weights is not None and t in weights:
                allocation = budget_for_buys * weights[t]
            else:
                allocation = budget_for_buys / len(buys)
            allocation = min(allocation, remaining_budget)
            if price > 0 and allocation > 0:
                net_investment = allocation / (1 + COST_RATE)
                cost = net_investment * COST_RATE
                shares[t] = net_investment / price
                total_cost += cost
                remaining_budget -= allocation

    held_value = sum(shares[t] * open_prices.get(t, 0.0) for t in shares)
    return equity_at_open - held_value - total_cost
