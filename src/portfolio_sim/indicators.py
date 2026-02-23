"""Kaufman's Adaptive Moving Average (KAMA) — Numba JIT accelerated.

Standalone implementation (no external config dependencies).
"""

import numpy as np
import pandas as pd
from numba import njit


@njit(cache=True)
def _kama_recurrent_loop(
    data_array: np.ndarray,
    smoothing_array: np.ndarray,
    kama: np.ndarray,
    period: int,
) -> np.ndarray:
    """Numba JIT recurrent KAMA update. cache=True for zero warm-up on restart."""
    n = len(data_array)
    for i in range(period + 1, n):
        smoothing = smoothing_array[i - period]
        kama[i] = kama[i - 1] + smoothing * (data_array[i] - kama[i - 1])
    return kama


def compute_kama(
    data: np.ndarray | pd.Series,
    period: int = 20,
    fast_constant: int = 2,
    slow_constant: int = 30,
) -> np.ndarray:
    """Compute Kaufman Adaptive Moving Average.

    Returns np.ndarray of KAMA values (NaN-padded at the start).
    """
    period = max(1, int(period))
    fast_constant = max(1, int(fast_constant))
    slow_constant = max(1, int(slow_constant))

    data_array = np.asarray(data, dtype=float)
    n = len(data_array)

    if n <= period:
        return np.full(n, np.nan, dtype=float)

    kama = np.full(n, np.nan, dtype=float)

    fast_const = 2.0 / (fast_constant + 1)
    slow_const = 2.0 / (slow_constant + 1)

    # Vectorized price change: abs(data[i] - data[i - period])
    price_change = np.abs(data_array[period:] - data_array[:-period])

    # Vectorized volatility: rolling sum of abs(diff(data)) over period
    abs_diff = np.abs(np.diff(data_array))
    abs_diff_cumsum = np.concatenate(([0], np.cumsum(abs_diff)))
    volatility = abs_diff_cumsum[period:] - abs_diff_cumsum[:-period]

    # Efficiency Ratio
    with np.errstate(divide="ignore", invalid="ignore"):
        er = np.where(volatility > 0, price_change / volatility, 0.0)
    er = np.clip(er, 0, 1)

    # Smoothing constant
    sc = (er * (fast_const - slow_const) + slow_const) ** 2
    smoothing_array = np.clip(sc, 0, 1)

    # Initial KAMA value
    kama[period] = data_array[period]

    # Recurrent loop via Numba JIT
    kama = _kama_recurrent_loop(data_array, smoothing_array, kama, period)

    return kama


def compute_kama_series(prices: pd.Series, period: int = 20) -> pd.Series:
    """Convenience wrapper returning pd.Series with the same index as input."""
    kama_arr = compute_kama(prices.values, period=period)
    return pd.Series(kama_arr, index=prices.index)
