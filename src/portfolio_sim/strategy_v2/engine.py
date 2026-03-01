"""V2 bar-by-bar simulation engine — KAMA momentum with vol-targeting overlay.

Extends the base engine with portfolio-level volatility targeting:
  - Tracks daily portfolio returns to estimate realised vol.
  - Scales position sizes by (target_vol / realised_vol) so the portfolio
    maintains a roughly constant annualised volatility.
  - When realised vol exceeds target, actively trims ALL positions
    proportionally to reduce exposure — this is the key difference vs v1's
    "lazy hold" approach.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.portfolio_sim.alpha import get_buy_candidates
from src.portfolio_sim.config import (
    ASSET_CLASS_MAP,
    COMMISSION_RATE,
    RISK_FREE_RATE,
    SLIPPAGE_RATE,
    SPY_TICKER,
)

# Asset classes treated as safe havens during risk-off (kept/buyable when SPY < KAMA)
_SAFE_HAVEN_CLASSES = frozenset({
    "Long Bonds", "Mid Bonds", "Short Bonds", "Corporate Bonds", "Metals",
})
from src.portfolio_sim.indicators import compute_kama_series
from src.portfolio_sim.models import SimulationResult
from src.portfolio_sim.strategy_v2.params import StrategyParamsV2

COST_RATE = COMMISSION_RATE + SLIPPAGE_RATE
DAILY_RF = (1 + RISK_FREE_RATE) ** (1 / 252) - 1


def run_simulation_v2(
    close_prices: pd.DataFrame,
    open_prices: pd.DataFrame,
    tickers: list[str],
    initial_capital: float,
    params: StrategyParamsV2 | None = None,
    kama_cache: dict[str, pd.Series] | None = None,
    show_progress: bool = False,
) -> SimulationResult:
    """Run a bar-by-bar portfolio simulation with vol-targeting overlay."""
    p = params or StrategyParamsV2()

    # ------------------------------------------------------------------
    # 0. Pre-compute KAMA for all tickers + SPY
    # ------------------------------------------------------------------
    if kama_cache is None:
        kama_cache = {}
        all_tickers = list(set(tickers) | {SPY_TICKER})
        ticker_iter = (
            tqdm(all_tickers, desc="Computing KAMA", unit="ticker")
            if show_progress
            else all_tickers
        )
        for t in ticker_iter:
            if t in close_prices.columns:
                kama_cache[t] = compute_kama_series(
                    close_prices[t].dropna(), period=p.kama_period
                )

    # ------------------------------------------------------------------
    # 0a. Pre-compute SPY KAMA for regime filter
    # ------------------------------------------------------------------
    spy_prices_full = close_prices[SPY_TICKER].dropna()
    spy_kama_series = compute_kama_series(spy_prices_full, period=p.kama_spy_period)

    # ------------------------------------------------------------------
    # 0b. Pre-compute vectorised alpha & volatility matrices
    # ------------------------------------------------------------------
    _ticker_cols = [t for t in tickers if t in close_prices.columns]
    _prices = close_prices[_ticker_cols]

    returns_df = _prices.pct_change()
    momentum_df = _prices / _prices.shift(p.lookback_period - 1) - 1.0

    _price_change = (_prices - _prices.shift(p.lookback_period)).abs()
    _daily_abs_diff = _prices.diff(1).abs()
    _volatility = _daily_abs_diff.rolling(p.lookback_period, min_periods=1).sum()
    er_df = (_price_change / _volatility).clip(0, 1).fillna(0)

    rp_vol_df = returns_df.rolling(p.volatility_lookback, min_periods=5).std()

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

    pending_trades: dict[str, float] | None = None
    pending_weights: dict[str, float] | None = None

    cash_values: list[float] = []
    holdings_rows: list[dict[str, float]] = []
    trade_log: list[dict] = []

    # V2: vol-targeting state
    portfolio_returns: list[float] = []
    current_scale: float = 1.0

    # ------------------------------------------------------------------
    # 3. Day-by-day loop
    # ------------------------------------------------------------------
    date_iter = (
        tqdm(sim_dates, desc="Simulating V2", unit="day")
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
            cash = _execute_trades_v2(
                shares,
                pending_trades,
                equity_at_open,
                daily_open,
                p.top_n,
                weights=pending_weights,
                scale=current_scale,
            )

            for t in shares_before - set(shares.keys()):
                trade_log.append(
                    {
                        "date": date,
                        "ticker": t,
                        "action": "sell",
                        "shares": 0.0,
                        "price": daily_open.get(t, 0.0),
                    }
                )
            for t in set(shares.keys()) - shares_before:
                trade_log.append(
                    {
                        "date": date,
                        "ticker": t,
                        "action": "buy",
                        "shares": shares[t],
                        "price": daily_open.get(t, 0.0),
                    }
                )
            # Log partial sells (vol-targeting trim)
            for t in shares_before & set(shares.keys()):
                old_shares = shares_before_counts.get(t, 0.0)
                if shares[t] < old_shares - 1e-10:
                    trade_log.append(
                        {
                            "date": date,
                            "ticker": t,
                            "action": "trim",
                            "shares": shares[t],
                            "price": daily_open.get(t, 0.0),
                        }
                    )
            pending_trades = None
            pending_weights = None

        # --- V2: active position reduction on Open ---
        # When vol is high and current_scale < 1, trim all positions
        # proportionally to match target exposure.
        if current_scale < 1.0 and shares:
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
                    # Need to trim: reduce each position proportionally
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
                            trade_log.append(
                                {
                                    "date": date,
                                    "ticker": t,
                                    "action": "trim",
                                    "shares": new_shares,
                                    "price": price,
                                }
                            )
                    held_value_new = sum(
                        shares[t] * daily_open.get(t, 0.0) for t in shares
                    )
                    cash = equity_at_open - held_value_new - trim_cost

        # --- Cash earns risk-free rate ---
        if cash > 0:
            cash *= 1 + DAILY_RF

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

        # --- V2: update portfolio return tracker ---
        if i > 0:
            prev_eq = equity_values[-2]
            if prev_eq > 0:
                portfolio_returns.append(equity_value / prev_eq - 1.0)

        # --- V2: compute vol-targeting scale for next day ---
        if len(portfolio_returns) >= p.portfolio_vol_lookback:
            recent_rets = portfolio_returns[-p.portfolio_vol_lookback :]
            realized_vol = np.std(recent_rets) * np.sqrt(252)
            if realized_vol > 1e-8:
                current_scale = p.target_vol / realized_vol
                current_scale = max(0.1, min(current_scale, p.max_leverage))
            else:
                current_scale = p.max_leverage
        # else: keep current_scale = 1.0 during warmup

        # --- Compute signals on Close(T) for Open(T+1) ---

        # SPY KAMA regime filter
        spy_kama_val = spy_kama_series.get(date, np.nan)
        spy_close_val = daily_close.get(SPY_TICKER, np.nan)
        risk_off = (
            not np.isnan(spy_kama_val)
            and not np.isnan(spy_close_val)
            and spy_close_val < spy_kama_val * (1 - p.kama_buffer)
        )

        sells: dict[str, float] = {}
        kama_row = (
            kama_df.loc[date]
            if (not kama_df.empty and date in kama_df.index)
            else pd.Series(dtype=float)
        )

        if risk_off:
            # Smart risk-off: sell risky assets, keep safe havens
            for t in list(shares.keys()):
                if ASSET_CLASS_MAP.get(t, "US Equity") not in _SAFE_HAVEN_CLASSES:
                    sells[t] = 0.0

            kama_current = kama_row.dropna().to_dict() if len(kama_row) > 0 else {}
            prices_row = _prices.loc[[date]]
            all_candidates = get_buy_candidates(
                prices_row, tickers, kama_current,
                kama_buffer=p.kama_buffer, top_n=p.top_n,
                use_risk_adjusted=p.use_risk_adjusted,
                precomputed_momentum=momentum_df.loc[date],
                precomputed_er=er_df.loc[date],
            )
            candidates = [
                c for c in all_candidates
                if ASSET_CLASS_MAP.get(c, "US Equity") in _SAFE_HAVEN_CLASSES
            ]
        else:
            kama_current = kama_row.dropna().to_dict() if len(kama_row) > 0 else {}
            prices_row = _prices.loc[[date]]

            # Extended candidate list for rotation stop (top_n * 2)
            extended_candidates = get_buy_candidates(
                prices_row, tickers, kama_current,
                kama_buffer=p.kama_buffer, top_n=p.top_n * 2,
                use_risk_adjusted=p.use_risk_adjusted,
                precomputed_momentum=momentum_df.loc[date],
                precomputed_er=er_df.loc[date],
            )
            candidates = extended_candidates[:p.top_n]

            for t in list(shares.keys()):
                t_kama = kama_row.get(t, np.nan)
                if not np.isnan(t_kama):
                    # Exit 1: KAMA trend break
                    if daily_close.get(t, 0.0) < t_kama * (1 - p.kama_buffer):
                        sells[t] = 0.0
                    # Exit 2: rotation stop — asset fell out of extended ranking
                    elif t not in extended_candidates:
                        sells[t] = 0.0

        # --- Correlation filter ---
        if p.corr_threshold < 1.0 and shares:
            held_after_sells = [t for t in shares if t not in sells]
            if held_after_sells:
                row_idx = warmup + i
                corr_start = max(0, row_idx - p.lookback_period + 1)
                recent_rets = returns_df.iloc[corr_start : row_idx + 1]

                all_corr_tickers = list(set(candidates) | set(held_after_sells))
                valid_cols = [
                    c for c in all_corr_tickers if c in recent_rets.columns
                ]

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

        # --- Build trade instructions ---
        new_trades: dict[str, float] = {}

        for t in sells:
            new_trades[t] = 0.0

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
                if p.weighting_mode == "risk_parity":
                    pending_weights = _compute_inverse_vol_weights_fast(
                        buys_list,
                        rp_vol_df.loc[date],
                    )
                else:
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


# ---------------------------------------------------------------------------
# Position sizing helpers
# ---------------------------------------------------------------------------
def _inv_vols_to_weights(
    inv_vols: dict[str, float],
    tickers_to_buy: list[str],
) -> dict[str, float]:
    """Normalise inverse-volatility values into portfolio weights."""
    if not inv_vols:
        return {t: 1.0 / len(tickers_to_buy) for t in tickers_to_buy}
    total = sum(inv_vols.values())
    return {t: v / total for t, v in inv_vols.items()}


_VOL_FLOOR = 0.06 / (252 ** 0.5)  # ~6% annualized floor for daily vol


def _compute_inverse_vol_weights_fast(
    tickers_to_buy: list[str],
    vol_row: pd.Series,
) -> dict[str, float]:
    """Compute inverse-volatility weights from a pre-computed volatility row.

    A volatility floor prevents ultra-low-vol assets (e.g. SGOV, SHV)
    from dominating the portfolio.
    """
    if not tickers_to_buy:
        return {}

    inv_vols: dict[str, float] = {}
    for t in tickers_to_buy:
        vol = vol_row.get(t, np.nan)
        if np.isnan(vol):
            continue
        vol = max(vol, _VOL_FLOOR)
        inv_vols[t] = 1.0 / vol

    return _inv_vols_to_weights(inv_vols, tickers_to_buy)


def _execute_trades_v2(
    shares: dict[str, float],
    trades: dict[str, float],
    equity_at_open: float,
    open_prices: pd.Series,
    top_n: int,
    weights: dict[str, float] | None = None,
    scale: float = 1.0,
) -> float:
    """Execute trades with vol-targeting scale.  Mutates ``shares``.

    The *scale* parameter controls the fraction of equity allocated to
    new buys.  When scale < 1.0 (high-vol regime), less capital is
    deployed; when scale > 1.0 (low-vol regime), up to *scale* fraction
    is deployed (capped by available cash — no margin).
    """
    total_cost = 0.0

    # Sells (identical to v1)
    for t, action in list(trades.items()):
        if action == 0.0 and t in shares:
            price = open_prices.get(t, 0.0)
            if price > 0:
                trade_value = shares[t] * price
                total_cost += trade_value * COST_RATE
            del shares[t]

    held_value = sum(shares[t] * open_prices.get(t, 0.0) for t in shares)
    available = equity_at_open - held_value - total_cost

    # V2: scale the budget for new buys
    target_invested = equity_at_open * min(scale, 1.0)  # cap at 100% for buys
    budget_for_buys = max(0.0, min(target_invested - held_value, available))

    buys = [
        t for t, action in trades.items() if action == 1.0 and t not in shares
    ]
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
