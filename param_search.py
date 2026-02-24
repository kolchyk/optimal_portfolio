"""Parameter grid search for maximizing 3-year profitability ratio vs SPY."""

import os
import itertools
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from src.portfolio_sim.engine import run_simulation
from src.portfolio_sim.params import StrategyParams
from src.portfolio_sim.reporting import compute_metrics

# ---------------------------------------------------------------------------
# Shared data for worker processes (initializer pattern avoids repeated
# pickling of large DataFrames — sent once per worker, not once per task).
# ---------------------------------------------------------------------------
_shared: dict = {}


def _init_worker(close: pd.DataFrame, opn: pd.DataFrame, tickers: list[str]):
    """Initializer for each worker process — stores shared data globally."""
    _shared["close"] = close
    _shared["open"] = opn
    _shared["tickers"] = tickers


def _evaluate_single(combo: tuple) -> dict:
    """Evaluate a single parameter combination using shared data."""
    kp, lb, tn, buf = combo
    close = _shared["close"]
    opn = _shared["open"]
    valid = _shared["tickers"]
    params = StrategyParams(
        kama_period=kp, lookback_period=lb, top_n=tn, kama_buffer=buf,
    )
    try:
        result = run_simulation(close, opn, valid, 10_000, params=params)
        end_date = result.equity.index[-1]
        three_years_ago = end_date - pd.Timedelta(days=3 * 365)

        strat_3y = result.equity[result.equity.index >= three_years_ago]
        spy_3y = result.spy_equity[result.spy_equity.index >= three_years_ago]

        strat_m = compute_metrics(strat_3y)
        spy_m = compute_metrics(spy_3y)

        ratio = strat_m["total_return"] / spy_m["total_return"] if spy_m["total_return"] > 0 else 0

        return {
            "kama_period": kp,
            "lookback_period": lb,
            "top_n": tn,
            "kama_buffer": buf,
            "strat_return_3y": strat_m["total_return"],
            "spy_return_3y": spy_m["total_return"],
            "ratio": ratio,
            "strat_cagr_3y": strat_m["cagr"],
            "strat_maxdd_3y": strat_m["max_drawdown"],
            "strat_sharpe_3y": strat_m["sharpe"],
        }
    except Exception as e:
        return {
            "kama_period": kp, "lookback_period": lb, "top_n": tn,
            "kama_buffer": buf, "ratio": -1, "error": str(e),
        }


def run_grid_search():
    close = pd.read_parquet("output/cache/close_prices.parquet")
    opn = pd.read_parquet("output/cache/open_prices.parquet")
    tickers = [t for t in close.columns if t != "SPY"]
    valid = [t for t in tickers if len(close[t].dropna()) >= 756]

    # Aggressive parameter grid focused on concentration + speed
    grid = {
        "kama_period": [5, 8, 10, 15],
        "lookback_period": [15, 20, 30, 40, 60],
        "top_n": [3, 5, 7, 10],
        "kama_buffer": [0.0, 0.003, 0.005, 0.01],
    }

    combos = list(itertools.product(
        grid["kama_period"], grid["lookback_period"],
        grid["top_n"], grid["kama_buffer"],
    ))
    n_combos = len(combos)
    n_workers = max(1, os.cpu_count() - 1)
    print(f"Testing {n_combos} combinations on {n_workers} workers...")

    results = []
    t0 = time.time()

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(close, opn, valid),
    ) as executor:
        futures = {executor.submit(_evaluate_single, c): c for c in combos}
        for future in tqdm(as_completed(futures), total=n_combos, desc="Grid search", unit="combo"):
            results.append(future.result())

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s")

    # Sort by ratio
    results = [r for r in results if r.get("ratio", -1) > 0]
    results.sort(key=lambda x: x["ratio"], reverse=True)

    print(f"\nTop 20 parameter combinations:")
    print(f"{'kama':>5} {'lbk':>4} {'top':>4} {'buf':>6} {'ratio':>7} {'ret_3y':>8} {'cagr':>7} {'maxdd':>7} {'sharpe':>7}")
    print("-" * 65)
    for r in results[:20]:
        print(f"{r['kama_period']:>5} {r['lookback_period']:>4} {r['top_n']:>4} "
              f"{r['kama_buffer']:>6.3f} {r['ratio']:>7.2f}x "
              f"{r.get('strat_return_3y', 0):>7.1%} "
              f"{r.get('strat_cagr_3y', 0):>7.1%} "
              f"{r.get('strat_maxdd_3y', 0):>7.1%} "
              f"{r.get('strat_sharpe_3y', 0):>7.2f}")


if __name__ == "__main__":
    run_grid_search()
