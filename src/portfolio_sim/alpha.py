"""Simple Baseline strategy — the production alpha.

5-step algorithm (Steps 2-5 here; Step 1 Market Breadth is in engine.py):
  2. KAMA trend filter   — keep only tickers with Close > KAMA
  3. Raw momentum ranking — score = Close[-1] / Close[-lookback] - 1
  4. Correlation walk-down — greedy selection, skip if corr > max_correlation
  5. Inverse volatility weighting — w ~ 1 / annualized_vol, cap at MAX_WEIGHT
"""

import numpy as np
import pandas as pd

from src.portfolio_sim.config import MAX_WEIGHT, SAFE_HAVEN_TICKER, VOL_FLOOR, StrategyParams


def compute_target_weights(
    prices_window: pd.DataFrame,
    tickers: list[str],
    params: StrategyParams,
    kama_values: dict[str, float],
) -> np.ndarray:
    """Compute target portfolio weights for the given price window.

    Args:
        prices_window: Close prices, rows = lookback_period trading days, cols = tickers.
        tickers: ordered list of ticker symbols matching engine's master list.
        params: the 4 tunable strategy parameters.
        kama_values: {ticker: current_kama_value} for KAMA entry filter.

    Returns:
        np.ndarray of weights aligned with *tickers*. Sums to <= 1.0.
        The caller (engine) allocates the remainder (1 - sum) to SHV.
    """
    # Step 2: KAMA trend filter
    candidates = [
        t
        for t in tickers
        if t != SAFE_HAVEN_TICKER
        and t in prices_window.columns
        and not np.isnan(prices_window[t].iloc[-1])
        and prices_window[t].iloc[-1] > kama_values.get(t, 0)
    ]

    if not candidates:
        return np.zeros(len(tickers))

    # Step 3: Raw 6-month momentum ranking
    momentum: dict[str, float] = {}
    for t in candidates:
        close_now = prices_window[t].iloc[-1]
        close_past = prices_window[t].iloc[0]
        if close_past > 1e-8:
            momentum[t] = close_now / close_past - 1
        else:
            momentum[t] = 0.0

    # Sort descending, keep only positive momentum
    ranked = sorted(
        [(t, s) for t, s in momentum.items() if s > 0],
        key=lambda x: x[1],
        reverse=True,
    )

    if not ranked:
        return np.zeros(len(tickers))

    # Step 4: Correlation walk-down
    candidate_tickers = [t for t, _ in ranked]
    returns_df = prices_window[candidate_tickers].pct_change().dropna()
    corr_matrix = returns_df.corr().fillna(0)

    selected: list[str] = []
    for t, _ in ranked:
        if len(selected) >= params.top_n_selection:
            break
        if not any(
            abs(corr_matrix.loc[t, s]) > params.max_correlation for s in selected
        ):
            selected.append(t)

    if not selected:
        return np.zeros(len(tickers))

    # Step 5: Inverse volatility weighting
    recent_returns = prices_window[selected].pct_change().iloc[-20:]
    vol = (recent_returns.std() * np.sqrt(252)).clip(lower=VOL_FLOOR)

    raw_weights = 1.0 / vol
    raw_weights = raw_weights / raw_weights.sum()

    # Cap at MAX_WEIGHT
    if len(selected) * MAX_WEIGHT < 1.0:
        # Not enough positions to fill 100% at MAX_WEIGHT — just clip.
        # Remainder will be allocated to SHV by the engine.
        raw_weights = raw_weights.clip(upper=MAX_WEIGHT)
    else:
        # Redistribute excess iteratively
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

    return result
