"""KAMA momentum alpha — Long/Cash only.

Selects buy candidates from the universe based on:
  1. KAMA trend filter: keep stocks where Close > KAMA * (1 + KAMA_BUFFER).
  2. Momentum ranking: sort by LOOKBACK_PERIOD-day return, take top N.

Does NOT dictate weights — sizing is handled by the engine.
"""

import numpy as np
import pandas as pd

from src.portfolio_sim.config import KAMA_BUFFER, TOP_N


def get_buy_candidates(
    prices_window: pd.DataFrame,
    tickers: list[str],
    kama_values: dict[str, float],
    kama_buffer: float = KAMA_BUFFER,
    top_n: int = TOP_N,
) -> list[str]:
    """Return an ordered list of top-momentum tickers passing the KAMA filter.

    Args:
        prices_window: Close prices with rows >= LOOKBACK_PERIOD trading days,
                       cols = tickers.
        tickers: ordered list of tradable ticker symbols (excludes SPY).
        kama_values: {ticker: current_kama_value} for KAMA filter.
        kama_buffer: hysteresis buffer for KAMA filter (default from config).
        top_n: maximum number of candidates to return (default from config).

    Returns:
        List of up to *top_n* ticker symbols, ranked by descending momentum.
        Empty list when no candidates pass both filters.
    """
    candidates = []

    for t in tickers:
        if t not in prices_window.columns:
            continue
        close = prices_window[t].iloc[-1]
        kama = kama_values.get(t, np.nan)
        if np.isnan(close) or np.isnan(kama):
            continue
        if close > kama * (1 + kama_buffer):
            candidates.append(t)

    if not candidates:
        return []

    momentum: dict[str, float] = {}
    for t in candidates:
        close_now = prices_window[t].iloc[-1]
        close_past = prices_window[t].iloc[0]
        if np.isnan(close_past) or close_past <= 1e-8:
            momentum[t] = 0.0
        else:
            momentum[t] = close_now / close_past - 1.0

    ranked = sorted(
        [(t, m) for t, m in momentum.items() if m > 0],
        key=lambda x: x[1],
        reverse=True,
    )

    return [t for t, _ in ranked[:top_n]]
