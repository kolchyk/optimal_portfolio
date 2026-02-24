"""Long/Short Equity alpha with Market Breathing regime.

3-step algorithm (Step 1 Market Breathing is in engine.py):
  2. KAMA trend filter   — Bull: keep Close > KAMA; Bear: keep Close < KAMA
  3. Momentum ranking    — Bull: strongest positive; Bear: weakest negative
  4. Inverse volatility weighting — w ~ 1/vol, cap at MAX_WEIGHT, negate for shorts
"""

import numpy as np
import pandas as pd

from src.portfolio_sim.config import (
    MAX_GROSS_EXPOSURE,
    MAX_WEIGHT,
    SAFE_HAVEN_TICKER,
    VOL_FLOOR,
    StrategyParams,
)


def compute_target_weights(
    prices_window: pd.DataFrame,
    tickers: list[str],
    params: StrategyParams,
    kama_values: dict[str, float],
    is_bull: bool = True,
) -> np.ndarray:
    """Compute target portfolio weights for the given price window.

    Args:
        prices_window: Close prices, rows = lookback_period trading days, cols = tickers.
        tickers: ordered list of ticker symbols matching engine's master list.
        params: the 3 tunable strategy parameters.
        kama_values: {ticker: current_kama_value} for KAMA filter.
        is_bull: True = long mode (SPY > KAMA), False = short mode (SPY < KAMA).

    Returns:
        np.ndarray of weights aligned with *tickers*.
        Bull mode: weights >= 0, sum <= 1.0.
        Bear mode: weights <= 0 for alpha positions.
    """
    # Step 2: KAMA trend filter (direction-aware)
    if is_bull:
        # Bull: keep tickers where Close > KAMA (uptrend)
        candidates = [
            t
            for t in tickers
            if t != SAFE_HAVEN_TICKER
            and t in prices_window.columns
            and not np.isnan(prices_window[t].iloc[-1])
            and prices_window[t].iloc[-1] > kama_values.get(t, np.inf)
        ]
    else:
        # Bear: keep tickers where Close < KAMA (downtrend)
        candidates = [
            t
            for t in tickers
            if t != SAFE_HAVEN_TICKER
            and t in prices_window.columns
            and not np.isnan(prices_window[t].iloc[-1])
            and prices_window[t].iloc[-1] < kama_values.get(t, -np.inf)
        ]

    if not candidates:
        return np.zeros(len(tickers))

    # Step 3: Momentum ranking (direction-aware)
    momentum: dict[str, float] = {}
    for t in candidates:
        close_now = prices_window[t].iloc[-1]
        close_past = prices_window[t].iloc[0]
        if close_past > 1e-8:
            momentum[t] = close_now / close_past - 1
        else:
            momentum[t] = 0.0

    if is_bull:
        # Keep only positive momentum, sort descending (strongest first)
        ranked = sorted(
            [(t, s) for t, s in momentum.items() if s > 0],
            key=lambda x: x[1],
            reverse=True,
        )
    else:
        # Keep only negative momentum, sort ascending (weakest first)
        ranked = sorted(
            [(t, s) for t, s in momentum.items() if s < 0],
            key=lambda x: x[1],
            reverse=False,
        )

    if not ranked:
        return np.zeros(len(tickers))

    # Select top N (replaces old correlation walk-down)
    selected = [t for t, _ in ranked[: params.top_n_selection]]

    # Step 4: Inverse volatility weighting
    recent_returns = prices_window[selected].pct_change(fill_method=None).iloc[-20:]
    vol = (recent_returns.std() * np.sqrt(252)).fillna(VOL_FLOOR).clip(lower=VOL_FLOOR)

    raw_weights = 1.0 / vol
    raw_weights = raw_weights / raw_weights.sum()

    # Cap at MAX_WEIGHT
    if len(selected) * MAX_WEIGHT < 1.0:
        raw_weights = raw_weights.clip(upper=MAX_WEIGHT)
    else:
        for _ in range(10):
            capped = raw_weights > MAX_WEIGHT
            if not capped.any():
                break
            excess = raw_weights[capped].sum() - capped.sum() * MAX_WEIGHT
            raw_weights[capped] = MAX_WEIGHT
            not_capped = ~capped
            if not_capped.any() and raw_weights[not_capped].sum() > 1e-8:
                raw_weights[not_capped] += (
                    raw_weights[not_capped] / raw_weights[not_capped].sum()
                ) * excess
            else:
                break

    # Map back to full ticker array
    result = np.zeros(len(tickers))
    for t in selected:
        idx = tickers.index(t)
        result[idx] = raw_weights[t]

    # Apply sign: negate weights for short mode
    if not is_bull:
        result = -result

    # Leverage control: ensure gross exposure <= MAX_GROSS_EXPOSURE
    gross = np.abs(result).sum()
    if gross > MAX_GROSS_EXPOSURE:
        result *= MAX_GROSS_EXPOSURE / gross

    return result
