"""KAMA momentum alpha — Long/Cash only.

Selects buy candidates from the universe based on:
  1. KAMA trend filter: keep stocks where Close > KAMA * (1 + KAMA_BUFFER).
  2. ER²-adjusted momentum ranking: sort by return * ER² (Cascade-style
     momentum), take top N.  Harshly penalizes choppy action, rewards
     smooth directional trends.

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
    precomputed_er: pd.Series | None = None,
) -> list[str]:
    """Return an ordered list of top-momentum tickers passing the KAMA filter.

    Args:
        prices_window: Close prices with rows >= LOOKBACK_PERIOD trading days,
                       cols = tickers.
        tickers: ordered list of tradable ticker symbols.
        kama_values: {ticker: current_kama_value} for KAMA filter.
        kama_buffer: hysteresis buffer for KAMA filter (default from config).
        top_n: maximum number of candidates to return (default from config).
        use_risk_adjusted: if True, rank by return * ER² instead of raw
                           return.  Harshly penalizes choppy price action.
        precomputed_momentum: optional pre-computed momentum (return) per ticker
                              for the current date. When provided, avoids
                              recomputing from prices_window.
        precomputed_er: optional pre-computed Efficiency Ratio per ticker
                        for the current date. Used with precomputed_momentum
                        for ER²-adjusted scoring.

    Returns:
        List of up to *top_n* ticker symbols, ranked by descending score.
        Empty list when no candidates pass the filters.
    """
    # --- Vectorized KAMA filter ---
    valid_tickers = [t for t in tickers if t in prices_window.columns]
    if not valid_tickers:
        return []

    close_row = prices_window[valid_tickers].iloc[-1]
    kama_s = pd.Series(kama_values).reindex(valid_tickers)
    mask = close_row.notna() & kama_s.notna() & (close_row > kama_s * (1 + kama_buffer))
    candidates = mask[mask].index.tolist()

    if not candidates:
        return []

    # --- Scoring ---
    if precomputed_momentum is not None:
        # Fast vectorized path (used during optimization)
        mom = precomputed_momentum.reindex(candidates)
        valid_mask = mom.notna() & (mom > 0)
        mom = mom[valid_mask]

        if mom.empty:
            return []

        if use_risk_adjusted and precomputed_er is not None:
            er = precomputed_er.reindex(mom.index)
            # fillna(1.0) so missing ER falls back to raw_return * 1.0² = raw_return
            scores = mom * er.fillna(1.0) ** 2
        else:
            scores = mom

        return scores.sort_values(ascending=False).head(top_n).index.tolist()
    else:
        # Fallback: per-ticker loop for non-precomputed path
        scores: dict[str, float] = {}
        for t in candidates:
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
                price_change = abs(close_now - close_past)
                daily_abs_diffs = series.diff().abs().iloc[1:]
                volatility_sum = daily_abs_diffs.sum()
                if volatility_sum > 1e-8:
                    er = min(price_change / volatility_sum, 1.0)
                else:
                    er = 0.0
                scores[t] = raw_return * (er ** 2)
            else:
                scores[t] = raw_return

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [t for t, _ in ranked[:top_n]]
