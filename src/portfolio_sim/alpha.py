"""KAMA momentum alpha — Long/Cash only.

Selects buy candidates from the universe based on:
  1. KAMA trend filter: keep stocks where Close > KAMA * (1 + KAMA_BUFFER).
  2. Risk-adjusted momentum ranking: sort by return/volatility (Sharpe-like
     momentum), take top N.  This prefers strong AND smooth uptrends.

Does NOT dictate weights — sizing is handled by the engine.
"""

import numpy as np
import pandas as pd

from src.portfolio_sim.config import (
    KAMA_BUFFER,
    TOP_N,
)


def get_buy_candidates(
    prices_window: pd.DataFrame,
    tickers: list[str],
    kama_values: dict[str, float],
    kama_buffer: float = KAMA_BUFFER,
    top_n: int = TOP_N,
    use_risk_adjusted: bool = True,
    precomputed_momentum: pd.Series | None = None,
    precomputed_vol: pd.Series | None = None,
) -> list[str]:
    """Return an ordered list of top-momentum tickers passing the KAMA filter.

    Args:
        prices_window: Close prices with rows >= LOOKBACK_PERIOD trading days,
                       cols = tickers.
        tickers: ordered list of tradable ticker symbols.
        kama_values: {ticker: current_kama_value} for KAMA filter.
        kama_buffer: hysteresis buffer for KAMA filter (default from config).
        top_n: maximum number of candidates to return (default from config).
        use_risk_adjusted: if True, rank by return/volatility instead of raw
                           return.  Prefers smooth uptrends.
        precomputed_momentum: optional pre-computed momentum (return) per ticker
                              for the current date. When provided, avoids
                              recomputing from prices_window.
        precomputed_vol: optional pre-computed rolling volatility per ticker
                         for the current date. Used with precomputed_momentum.

    Returns:
        List of up to *top_n* ticker symbols, ranked by descending score.
        Empty list when no candidates pass the filters.
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

    scores: dict[str, float] = {}
    use_precomputed = precomputed_momentum is not None

    for t in candidates:
        if use_precomputed:
            raw_return = precomputed_momentum.get(t, np.nan)
            if np.isnan(raw_return) or raw_return <= 0:
                continue
            if use_risk_adjusted and precomputed_vol is not None:
                vol = precomputed_vol.get(t, np.nan)
                if not np.isnan(vol) and vol > 1e-8:
                    scores[t] = raw_return / vol
                else:
                    scores[t] = raw_return
            else:
                scores[t] = raw_return
        else:
            series = prices_window[t].dropna()
            if len(series) < 5:
                continue
            close_now = series.iloc[-1]
            close_past = series.iloc[0]
            if np.isnan(close_past) or close_past <= 1e-8:
                continue
            raw_return = close_now / close_past - 1.0
            if raw_return <= 0:
                continue

            if use_risk_adjusted:
                daily_returns = series.pct_change().dropna()
                vol = daily_returns.std()
                if vol > 1e-8:
                    scores[t] = raw_return / vol
                else:
                    scores[t] = raw_return
            else:
                scores[t] = raw_return

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ranked_tickers = [t for t, _ in ranked]

    return ranked_tickers[:top_n]
