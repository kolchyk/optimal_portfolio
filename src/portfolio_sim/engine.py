"""Bar-by-bar simulation engine — Long/Cash only.

Signals computed on Close(T), execution on Open(T+1).

Key design decisions that prevent capital destruction:
  - Lazy hold: positions are sold ONLY when their own KAMA stop-loss triggers,
    never because they dropped in the momentum ranking.
  - Position sizing: inverse-volatility (risk parity) — low-vol assets get
    larger allocations so each position contributes roughly equal dollar risk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.portfolio_sim.alpha import get_buy_candidates
from src.portfolio_sim.config import (
    COMMISSION_RATE,
    RISK_FREE_RATE,
    SLIPPAGE_RATE,
    SPY_TICKER,
)
from src.portfolio_sim.indicators import compute_kama_series
from src.portfolio_sim.models import SimulationResult
from src.portfolio_sim.params import StrategyParams

COST_RATE = COMMISSION_RATE + SLIPPAGE_RATE
DAILY_RF = (1 + RISK_FREE_RATE) ** (1 / 252) - 1


def run_simulation(
    close_prices: pd.DataFrame,
    open_prices: pd.DataFrame,
    tickers: list[str],
    initial_capital: float,
    params: StrategyParams | None = None,
    kama_cache: dict[str, pd.Series] | None = None,
    spy_kama_series: pd.Series | None = None,
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
    # 0a. Pre-compute SPY KAMA for regime filter
    # ------------------------------------------------------------------
    if spy_kama_series is None:
        spy_prices = close_prices[SPY_TICKER].dropna()
        spy_kama_series = compute_kama_series(spy_prices, period=p.kama_spy_period)

    # ------------------------------------------------------------------
    # 0b. Pre-compute vectorized alpha & volatility matrices
    # ------------------------------------------------------------------
    # Daily returns, rolling volatility, and momentum — computed once here,
    # then looked up via O(1) .loc[date] inside the daily loop instead of
    # recalculating pct_change() / std() for every ticker on every day.
    _ticker_cols = [t for t in tickers if t in close_prices.columns]
    _prices = close_prices[_ticker_cols]

    returns_df = _prices.pct_change()
    momentum_df = _prices / _prices.shift(p.lookback_period - 1) - 1.0

    # Efficiency Ratio matrix: vectorized across all tickers at once
    _price_change = (_prices - _prices.shift(p.lookback_period)).abs()
    _daily_abs_diff = _prices.diff(1).abs()
    _volatility = _daily_abs_diff.rolling(p.lookback_period, min_periods=1).sum()
    er_df = (_price_change / _volatility).clip(0, 1).fillna(0)

    rp_vol_df = returns_df.rolling(p.volatility_lookback, min_periods=5).std()

    # Pre-compute KAMA DataFrame for fast vectorized row lookups.
    # Converts {ticker: Series} dict → single DataFrame (tickers as columns).
    kama_df = pd.DataFrame(kama_cache) if kama_cache else pd.DataFrame()

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
                p.top_n, weights=pending_weights,
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

        # --- Cash earns risk-free rate (T-bill proxy) ---
        if cash > 0:
            cash *= 1 + DAILY_RF

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

        # SPY KAMA regime filter: when SPY < KAMA, sell all and skip buys
        spy_kama_val = spy_kama_series.get(date, np.nan) if spy_kama_series is not None else np.nan
        spy_close_val = daily_close.get(SPY_TICKER, np.nan)
        risk_off = (
            not np.isnan(spy_kama_val)
            and not np.isnan(spy_close_val)
            and spy_close_val < spy_kama_val * (1 - p.kama_buffer)
        )

        sells: dict[str, float] = {}
        kama_row = kama_df.loc[date] if (not kama_df.empty and date in kama_df.index) else pd.Series(dtype=float)

        if risk_off:
            # Risk-off: liquidate all positions, no new buys
            for t in list(shares.keys()):
                sells[t] = 0.0
            candidates: list[str] = []
        else:
            # Sell ONLY on individual KAMA stop-loss.
            # A stock dropping from rank 20 to rank 50 is irrelevant
            # as long as its own trend (Close > KAMA) holds.
            for t in list(shares.keys()):
                t_kama = kama_row.get(t, np.nan)
                if not np.isnan(t_kama):
                    if daily_close.get(t, 0.0) < t_kama * (1 - p.kama_buffer):
                        sells[t] = 0.0

            # Get fresh candidates from alpha (using pre-computed matrices)
            kama_current = kama_row.dropna().to_dict() if len(kama_row) > 0 else {}

            # Single-row DataFrame for the KAMA filter (Close > KAMA check)
            prices_row = _prices.loc[[date]]

            candidates = get_buy_candidates(
                prices_row, tickers, kama_current,
                kama_buffer=p.kama_buffer, top_n=p.top_n,
                use_risk_adjusted=p.use_risk_adjusted,
                precomputed_momentum=momentum_df.loc[date],
                precomputed_er=er_df.loc[date],
            )

        # --- Correlation filter: skip candidates too correlated with holdings ---
        if p.corr_threshold < 1.0 and shares:
            held_after_sells = [t for t in shares if t not in sells]
            if held_after_sells:
                # Use recent lookback_period rows for correlation
                row_idx = warmup + i
                corr_start = max(0, row_idx - p.lookback_period + 1)
                recent_rets = returns_df.iloc[corr_start:row_idx + 1]

                # Compute ONE correlation matrix for all relevant tickers
                all_corr_tickers = list(set(candidates) | set(held_after_sells))
                valid_cols = [c for c in all_corr_tickers if c in recent_rets.columns]

                if valid_cols:
                    corr_matrix = recent_rets[valid_cols].corr(min_periods=5)

                    filtered: list[str] = []
                    for t in candidates:
                        if t not in corr_matrix.columns:
                            filtered.append(t)
                            continue
                        skip = False
                        for h in held_after_sells:
                            if h not in corr_matrix.columns:
                                continue
                            corr_val = corr_matrix.at[t, h]
                            if np.isnan(corr_val):
                                continue
                            if corr_val > p.corr_threshold:
                                skip = True
                                break
                        if not skip:
                            filtered.append(t)
                    candidates = filtered

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

            # Compute weights for new buys
            buys_list = [t for t, v in new_trades.items() if v == 1.0]
            if buys_list:
                if p.weighting_mode == "risk_parity":
                    pending_weights = _compute_inverse_vol_weights_fast(
                        buys_list, rp_vol_df.loc[date],
                    )
                else:
                    # Equal weight across new buys
                    w = 1.0 / len(buys_list)
                    pending_weights = {t: w for t in buys_list}

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


def _inv_vols_to_weights(
    inv_vols: dict[str, float],
    tickers_to_buy: list[str],
) -> dict[str, float]:
    """Normalize inverse-volatility values into portfolio weights.

    Falls back to equal weight when no valid inverse-volatility could be
    computed for any ticker.
    """
    if not inv_vols:
        return {t: 1.0 / len(tickers_to_buy) for t in tickers_to_buy}

    total = sum(inv_vols.values())
    return {t: v / total for t, v in inv_vols.items()}


def _compute_inverse_vol_weights_fast(
    tickers_to_buy: list[str],
    vol_row: pd.Series,
) -> dict[str, float]:
    """Compute inverse-volatility weights from a pre-computed volatility row.

    Each ticker's weight is proportional to 1/volatility, so low-vol assets
    get larger allocations and high-vol assets get smaller allocations.
    """
    if not tickers_to_buy:
        return {}

    inv_vols: dict[str, float] = {}
    for t in tickers_to_buy:
        vol = vol_row.get(t, np.nan)
        if np.isnan(vol):
            continue
        if vol > 1e-8:
            inv_vols[t] = 1.0 / vol
        else:
            inv_vols[t] = 1.0  # near-zero vol: treat as equal weight

    return _inv_vols_to_weights(inv_vols, tickers_to_buy)


def _execute_trades(
    shares: dict[str, float],
    trades: dict[str, float],
    equity_at_open: float,
    open_prices: pd.Series,
    top_n: int = StrategyParams().top_n,
    weights: dict[str, float] | None = None,
) -> float:
    """Execute trades. Mutates ``shares`` in place. Returns remaining cash.

    Convention:
      - trades[t] == 0.0: sell position t.
      - trades[t] == 1.0: buy position t.

    When *weights* is provided (risk parity mode), each buy's allocation is
    equity_at_open * weights[t]. Otherwise strict 1/top_n equal weight.
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

    # Position sizing: distribute available cash proportionally among new buys.
    buys = [t for t, action in trades.items() if action == 1.0 and t not in shares]
    if buys and available > 0:
        available_for_buys = available  # snapshot before buy loop
        for t in buys:
            price = open_prices.get(t, 0.0)
            if weights is not None and t in weights:
                max_allocation = available_for_buys * weights[t]
            else:
                max_allocation = available_for_buys / len(buys)
            allocation = min(max_allocation, available)
            if price > 0 and allocation > 0:
                net_investment = allocation / (1 + COST_RATE)
                cost = net_investment * COST_RATE
                shares[t] = net_investment / price
                total_cost += cost
                available -= allocation

    held_value = sum(shares[t] * open_prices.get(t, 0.0) for t in shares)
    return equity_at_open - held_value - total_cost
