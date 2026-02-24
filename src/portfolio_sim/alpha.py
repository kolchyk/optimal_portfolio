"""KAMA momentum alpha — Long/Cash only, equal weight.

Algorithm (Market Breathing check is done in engine.py):
  1. KAMA trend filter: keep stocks where Close > KAMA * (1 + KAMA_BUFFER).
  2. Momentum ranking: sort by LOOKBACK_PERIOD-day return, take top N.
  3. Equal weight: 1/TOP_N per selected stock.
"""

import numpy as np
import pandas as pd

from src.portfolio_sim.config import KAMA_BUFFER, TOP_N


def compute_target_weights(
    prices_window: pd.DataFrame,
    tickers: list[str],
    kama_values: dict[str, float],
    is_bull: bool = True,
) -> dict[str, float]:
    """Compute equal-weight target portfolio for selected momentum stocks.

    Args:
        prices_window: Close prices with rows >= LOOKBACK_PERIOD trading days,
                       cols = tickers.
        tickers: ordered list of tradable ticker symbols (excludes SPY).
        kama_values: {ticker: current_kama_value} for KAMA filter.
        is_bull: True = SPY above KAMA (risk-on), False = risk-off.

    Returns:
        dict mapping ticker -> target weight (0.0 to 1/TOP_N).
        If bear regime or no candidates, returns empty dict (100% cash).
    """
    if not is_bull:
        return {}

    # KAMA trend filter with buffer
    candidates = []
    for t in tickers:
        if t not in prices_window.columns:
            continue
        close = prices_window[t].iloc[-1]
        kama = kama_values.get(t, np.nan)
        if np.isnan(close) or np.isnan(kama):
            continue
        if close > kama * (1 + KAMA_BUFFER):
            candidates.append(t)

    if not candidates:
        return {}

    # Momentum ranking (period-day return)
    momentum: dict[str, float] = {}
    for t in candidates:
        close_now = prices_window[t].iloc[-1]
        close_past = prices_window[t].iloc[0]
        if close_past > 1e-8:
            momentum[t] = close_now / close_past - 1.0
        else:
            momentum[t] = 0.0

    # Keep only positive momentum, sort descending, take top N
    ranked = sorted(
        [(t, m) for t, m in momentum.items() if m > 0],
        key=lambda x: x[1],
        reverse=True,
    )

    if not ranked:
        return {}

    selected = [t for t, _ in ranked[:TOP_N]]

    # Equal weight
    weight = 1.0 / TOP_N
    return {t: weight for t in selected}
