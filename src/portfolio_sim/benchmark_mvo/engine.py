"""Markowitz Mean-Variance Optimization benchmark strategy.

Implements three MVO variants for comparison with R² Momentum:
  - Min-Variance Portfolio (default, most robust)
  - Max-Sharpe (tangency) Portfolio
  - Risk Parity (equal risk contribution)

Uses scipy.optimize.minimize (SLSQP) for portfolio optimization.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.portfolio_sim.benchmark_mvo.params import MVOParams
from src.portfolio_sim.config import (
    COMMISSION_RATE,
    INITIAL_CAPITAL,
    RISK_FREE_RATE,
    SLIPPAGE_RATE,
    SPY_TICKER,
)

COST_RATE = COMMISSION_RATE + SLIPPAGE_RATE
DAILY_RF = (1 + RISK_FREE_RATE) ** (1 / 252) - 1


# ---------------------------------------------------------------------------
# Portfolio optimization solvers
# ---------------------------------------------------------------------------

def compute_min_variance_weights(
    cov_matrix: np.ndarray,
    max_weight: float = 0.20,
) -> np.ndarray:
    """Solve for minimum-variance portfolio weights.

    min  w^T * Cov * w
    s.t. sum(w) = 1, 0 <= w_i <= max_weight
    """
    n = cov_matrix.shape[0]
    w0 = np.ones(n) / n

    def objective(w: np.ndarray) -> float:
        return float(w @ cov_matrix @ w)

    result = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=[(0.0, max_weight)] * n,
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    if result.success:
        weights = np.maximum(result.x, 0.0)
        return weights / weights.sum()
    return np.ones(n) / n


def compute_max_sharpe_weights(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    max_weight: float = 0.20,
) -> np.ndarray:
    """Solve for maximum-Sharpe (tangency) portfolio weights."""
    n = cov_matrix.shape[0]
    w0 = np.ones(n) / n
    annual_rf = RISK_FREE_RATE

    def neg_sharpe(w: np.ndarray) -> float:
        port_ret = float(w @ expected_returns)
        port_vol = float(np.sqrt(w @ cov_matrix @ w))
        if port_vol < 1e-10:
            return 0.0
        return -(port_ret - annual_rf) / port_vol

    result = minimize(
        neg_sharpe,
        w0,
        method="SLSQP",
        bounds=[(0.0, max_weight)] * n,
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    if result.success:
        weights = np.maximum(result.x, 0.0)
        return weights / weights.sum()
    return np.ones(n) / n


def compute_risk_parity_weights(
    cov_matrix: np.ndarray,
    max_weight: float = 0.20,
) -> np.ndarray:
    """Solve for risk parity (equal risk contribution) weights."""
    n = cov_matrix.shape[0]
    w0 = np.ones(n) / n

    def objective(w: np.ndarray) -> float:
        port_vol = float(np.sqrt(w @ cov_matrix @ w))
        if port_vol < 1e-10:
            return 0.0
        marginal = cov_matrix @ w
        risk_contrib = w * marginal / port_vol
        target = port_vol / n
        return float(np.sum((risk_contrib - target) ** 2))

    result = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=[(0.0, max_weight)] * n,
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    if result.success:
        weights = np.maximum(result.x, 0.0)
        return weights / weights.sum()
    return np.ones(n) / n


# ---------------------------------------------------------------------------
# Portfolio optimization dispatcher
# ---------------------------------------------------------------------------

def optimize_portfolio(
    returns_window: pd.DataFrame,
    params: MVOParams,
) -> dict[str, float]:
    """Run MVO optimization on a window of daily returns.

    Returns dict mapping ticker -> weight (only weights > 0.1%).
    """
    clean = returns_window.dropna(axis=1)
    if clean.shape[1] < 2:
        if clean.shape[1] == 1:
            return {clean.columns[0]: 1.0}
        return {}

    tickers = clean.columns.tolist()
    cov_matrix = clean.cov().values * 252  # annualized

    if params.objective == "min_variance":
        weights = compute_min_variance_weights(cov_matrix, params.max_weight)
    elif params.objective == "max_sharpe":
        expected_returns = clean.mean().values * 252
        weights = compute_max_sharpe_weights(
            expected_returns, cov_matrix, params.max_weight,
        )
    elif params.objective == "risk_parity":
        weights = compute_risk_parity_weights(cov_matrix, params.max_weight)
    else:
        msg = f"Unknown objective: {params.objective}"
        raise ValueError(msg)

    result = {t: float(w) for t, w in zip(tickers, weights) if w > 0.001}
    total = sum(result.values())
    if total > 0:
        result = {t: w / total for t, w in result.items()}
    return result


# ---------------------------------------------------------------------------
# Rebalance helpers
# ---------------------------------------------------------------------------

def _get_rebalance_dates(
    dates: pd.DatetimeIndex,
    freq: str,
) -> set[pd.Timestamp]:
    """First trading day of each month or quarter."""
    rebal: set[pd.Timestamp] = set()
    seen: set[tuple[int, int]] = set()

    for date in dates:
        if freq == "month":
            key = (date.year, date.month)
        elif freq == "quarter":
            key = (date.year, (date.month - 1) // 3)
        else:
            msg = f"Unknown rebal_freq: {freq}"
            raise ValueError(msg)

        if key not in seen:
            seen.add(key)
            rebal.add(date)
    return rebal


def _get_returns_window(
    close_prices: pd.DataFrame,
    tickers: list[str],
    date: pd.Timestamp,
    params: MVOParams,
) -> pd.DataFrame | None:
    """Extract daily returns window for covariance estimation."""
    cols = [t for t in tickers if t in close_prices.columns]
    prices_up_to = close_prices.loc[:date, cols]
    if len(prices_up_to) < params.cov_lookback + 1:
        return None
    window = prices_up_to.iloc[-(params.cov_lookback + 1) :]
    returns = window.pct_change(fill_method=None).iloc[1:]
    threshold = len(returns) * 0.1
    valid_cols = [c for c in returns.columns if returns[c].isna().sum() < threshold]
    if not valid_cols:
        return None
    return returns[valid_cols].fillna(0)


def _execute_rebalance(
    shares: dict[str, float],
    target_weights: dict[str, float],
    cash: float,
    open_prices: pd.Series,
    trade_log: list[dict],
    date: pd.Timestamp,
) -> float:
    """Rebalance portfolio to target weights (full sell-then-buy).

    Mutates ``shares`` in place. Returns new cash balance.
    """
    held_value = sum(shares[t] * open_prices.get(t, 0.0) for t in shares)
    total_equity = cash + held_value
    if total_equity <= 0:
        return cash

    total_cost = 0.0

    # Sell all current positions
    for t in list(shares.keys()):
        price = open_prices.get(t, 0.0)
        if price > 0 and shares[t] > 0:
            trade_value = shares[t] * price
            total_cost += trade_value * COST_RATE
            trade_log.append({
                "date": date, "ticker": t, "action": "sell",
                "shares": shares[t], "price": price,
            })
        del shares[t]

    available = total_equity - total_cost

    # Buy target portfolio
    for t, w in target_weights.items():
        price = open_prices.get(t, np.nan)
        if np.isnan(price) or price <= 0:
            continue
        allocation = available * w
        if allocation > 0:
            net_investment = allocation / (1 + COST_RATE)
            cost = net_investment * COST_RATE
            shares[t] = net_investment / price
            total_cost += cost
            trade_log.append({
                "date": date, "ticker": t, "action": "buy",
                "shares": shares[t], "price": price,
            })

    new_held = sum(shares[t] * open_prices.get(t, 0.0) for t in shares)
    return total_equity - new_held - total_cost


# ---------------------------------------------------------------------------
# Main backtest
# ---------------------------------------------------------------------------

def run_mvo_backtest(
    close_prices: pd.DataFrame,
    open_prices: pd.DataFrame,
    tickers: list[str],
    initial_capital: float = INITIAL_CAPITAL,
    params: MVOParams | None = None,
) -> tuple[pd.Series, pd.Series, pd.DataFrame, pd.Series, list[dict]]:
    """Run MVO benchmark backtest.

    Returns:
        (equity, spy_equity, holdings_history, cash_history, trade_log)
        Same format as engine.run_backtest for compatibility.
    """
    p = params or MVOParams()
    warmup = p.warmup

    sim_dates = close_prices.index[warmup:]
    if len(sim_dates) == 0:
        empty = pd.Series(dtype=float)
        return empty, empty, pd.DataFrame(dtype=float), empty, []

    rebal_dates = _get_rebalance_dates(sim_dates, p.rebal_freq)

    cash = initial_capital
    shares: dict[str, float] = {}
    equity_values: list[float] = []
    cash_values: list[float] = []
    holdings_snapshots: list[dict[str, float]] = []
    trade_log: list[dict] = []

    pending_weights: dict[str, float] | None = None
    execute_on_next_open = False

    for date in sim_dates:
        daily_close = close_prices.loc[date]
        daily_open = open_prices.loc[date]

        # Execute pending trades on Open
        if execute_on_next_open and pending_weights is not None:
            cash = _execute_rebalance(
                shares, pending_weights, cash, daily_open, trade_log, date,
            )
            execute_on_next_open = False
            pending_weights = None

        # Cash earns risk-free rate
        if cash > 0:
            cash *= 1 + DAILY_RF

        # Mark-to-market on Close
        equity = cash + sum(
            shares.get(t, 0) * daily_close.get(t, 0.0) for t in shares
        )
        equity_values.append(equity)
        cash_values.append(cash)
        holdings_snapshots.append(dict(shares))

        if equity <= 0:
            remaining = len(sim_dates) - len(equity_values)
            equity_values.extend([0.0] * remaining)
            cash_values.extend([0.0] * remaining)
            holdings_snapshots.extend([{} for _ in range(remaining)])
            break

        # On rebalance date: compute new target weights
        if date in rebal_dates:
            returns_window = _get_returns_window(close_prices, tickers, date, p)
            if returns_window is not None and len(returns_window) >= p.min_history:
                target_weights = optimize_portfolio(returns_window, p)
                if target_weights:
                    pending_weights = target_weights
                    execute_on_next_open = True

    # Build output
    n = len(equity_values)
    equity_series = pd.Series(equity_values, index=sim_dates[:n])

    spy_close = close_prices[SPY_TICKER].reindex(sim_dates[:n]).ffill()
    if not spy_close.empty and spy_close.iloc[0] > 0:
        spy_equity = initial_capital * (spy_close / spy_close.iloc[0])
    else:
        spy_equity = pd.Series(initial_capital, index=sim_dates[:n])

    all_held_tickers = sorted({t for snap in holdings_snapshots for t in snap})
    holdings_data = {
        t: [snap.get(t, 0.0) for snap in holdings_snapshots]
        for t in all_held_tickers
    }
    holdings_history = (
        pd.DataFrame(holdings_data, index=sim_dates[:n])
        if all_held_tickers
        else pd.DataFrame(index=sim_dates[:n])
    )

    cash_history = pd.Series(cash_values, index=sim_dates[:n])
    return equity_series, spy_equity, holdings_history, cash_history, trade_log
