"""Quick backtest runner for fast strategy iteration.

Usage: python quick_backtest.py
"""

import pandas as pd
import numpy as np
import time

from src.portfolio_sim.cli_utils import filter_valid_tickers
from src.portfolio_sim.config import (
    INITIAL_CAPITAL,
    KAMA_BUFFER,
    KAMA_PERIOD,
    LOOKBACK_PERIOD,
    TOP_N,
)
from src.portfolio_sim.data import fetch_price_data, fetch_etf_tickers
from src.portfolio_sim.engine import run_simulation
from src.portfolio_sim.params import StrategyParams
from src.portfolio_sim.reporting import compute_metrics


def run_quick_backtest(
    kama_period: int = KAMA_PERIOD,
    lookback_period: int = LOOKBACK_PERIOD,
    top_n: int = TOP_N,
    kama_buffer: float = KAMA_BUFFER,
    initial_capital: float = INITIAL_CAPITAL,
    verbose: bool = True,
) -> dict:
    """Run backtest and return metrics for strategy and SPY over last 3 years."""
    # Load cached data
    try:
        close = pd.read_parquet("output/cache/close_prices_etf.parquet")
        opn = pd.read_parquet("output/cache/open_prices_etf.parquet")
    except FileNotFoundError:
        # Fallback if no etf cache yet
        tickers = fetch_etf_tickers()
        close, opn = fetch_price_data(tickers, period="5y", cache_suffix="_etf")

    min_days = 756
    valid = filter_valid_tickers(close, min_days)

    params = StrategyParams(
        kama_period=kama_period,
        lookback_period=lookback_period,
        top_n=top_n,
        kama_buffer=kama_buffer,
    )

    t0 = time.time()
    result = run_simulation(close, opn, valid, initial_capital, params=params)
    elapsed = time.time() - t0

    # Full period metrics
    strat_m = compute_metrics(result.equity)
    spy_m = compute_metrics(result.spy_equity)

    # Last 3 years metrics
    end_date = result.equity.index[-1]
    three_years_ago = end_date - pd.Timedelta(days=3 * 365)

    strat_3y = result.equity[result.equity.index >= three_years_ago]
    spy_3y = result.spy_equity[result.spy_equity.index >= three_years_ago]

    strat_3y_m = compute_metrics(strat_3y)
    spy_3y_m = compute_metrics(spy_3y)

    ratio = strat_3y_m["total_return"] / spy_3y_m["total_return"] if spy_3y_m["total_return"] > 0 else 0

    if verbose:
        print(f"\n{'='*60}")
        print(f"BACKTEST RESULTS (elapsed: {elapsed:.1f}s)")
        print(f"{'='*60}")
        print(f"Params: kama={kama_period}, lookback={lookback_period}, "
              f"top_n={top_n}, buffer={kama_buffer}")
        print(f"\nFull period ({result.equity.index[0].date()} to {end_date.date()}):")
        print(f"  Strategy:  CAGR={strat_m['cagr']:.1%}  Return={strat_m['total_return']:.1%}  "
              f"MaxDD={strat_m['max_drawdown']:.1%}  Sharpe={strat_m['sharpe']:.2f}")
        print(f"  S&P 500:   CAGR={spy_m['cagr']:.1%}  Return={spy_m['total_return']:.1%}  "
              f"MaxDD={spy_m['max_drawdown']:.1%}  Sharpe={spy_m['sharpe']:.2f}")
        print(f"\nLast 3 years ({strat_3y.index[0].date()} to {end_date.date()}):")
        print(f"  Strategy:  CAGR={strat_3y_m['cagr']:.1%}  Return={strat_3y_m['total_return']:.1%}  "
              f"MaxDD={strat_3y_m['max_drawdown']:.1%}  Sharpe={strat_3y_m['sharpe']:.2f}")
        print(f"  S&P 500:   CAGR={spy_3y_m['cagr']:.1%}  Return={spy_3y_m['total_return']:.1%}  "
              f"MaxDD={spy_3y_m['max_drawdown']:.1%}  Sharpe={spy_3y_m['sharpe']:.2f}")
        print(f"\n  >>> PROFITABILITY RATIO (Strategy/SPY): {ratio:.2f}x <<<")
        target = "ACHIEVED" if ratio >= 4.0 else f"NEED {4.0 - ratio:.2f}x MORE"
        print(f"  >>> Target 4.0x: {target}")
        print(f"{'='*60}")

    return {
        "strat_3y": strat_3y_m,
        "spy_3y": spy_3y_m,
        "ratio": ratio,
        "strat_full": strat_m,
        "spy_full": spy_m,
        "elapsed": elapsed,
        "result": result,
    }


if __name__ == "__main__":
    run_quick_backtest()
