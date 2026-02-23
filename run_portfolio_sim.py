"""CLI entry point for portfolio simulation.

Usage:
    python run_portfolio_sim.py --walk-forward --n-trials 100
    python run_portfolio_sim.py  # single run with default params
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import structlog

from src.portfolio_sim.config import INITIAL_CAPITAL, StrategyParams
from src.portfolio_sim.data import fetch_price_data, load_tickers
from src.portfolio_sim.engine import run_simulation
from src.portfolio_sim.reporting import (
    compute_metrics,
    format_metrics_table,
    save_wfv_json,
    save_wfv_report,
)
from src.portfolio_sim.walk_forward import run_walk_forward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Portfolio Simulation CLI")
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        default=False,
        help="Run Walk-Forward Validation (production mode)",
    )
    parser.add_argument(
        "--metric",
        choices=["calmar", "sharpe", "return"],
        default="calmar",
        help="Optimization metric (default: calmar)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of Optuna trials per WFV window (default: 100)",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force refresh data cache from yfinance",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Configure structlog
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.stdlib.LoggerFactory(),
    )

    # Output directory
    mode = "wfv" if args.walk_forward else "sim"
    dt = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output") / f"{mode}_{args.metric}_{args.n_trials}t_{dt}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load data
    print("\nLoading ticker universe...")
    tickers_600, original_portfolio = load_tickers()

    full_list = list(set(tickers_600 + list(original_portfolio.keys())))
    print(f"Fetching prices for {len(full_list)} tickers...")
    close_prices, open_prices = fetch_price_data(full_list, refresh=args.refresh)

    # Filter tickers with sufficient history (>= 3 years)
    min_len = 756  # ~3 years of trading days
    valid_tickers = [
        t for t in close_prices.columns if len(close_prices[t].dropna()) >= min_len
    ]
    close_prices = close_prices[valid_tickers].ffill().bfill()
    open_prices = open_prices[valid_tickers].ffill().bfill()

    all_tickers = [t for t in valid_tickers if t != "SPY"]
    print(f"Tradable tickers: {len(all_tickers)}")

    if args.walk_forward:
        # Walk-Forward Validation (production mode)
        wfv_result = run_walk_forward(
            close_prices,
            open_prices,
            all_tickers,
            n_trials=args.n_trials,
            metric=args.metric,
        )

        save_wfv_report(wfv_result, metric=args.metric, output_dir=output_dir)
        save_wfv_json(wfv_result, output_dir=output_dir)

        # Print final OOS metrics
        oos_metrics = compute_metrics(wfv_result["oos_equity"])
        print(f"\n{format_metrics_table(oos_metrics)}")

    else:
        # Single run with default params
        params = StrategyParams()
        print(f"\nSingle run with default params: {params}")

        # Use last ~3 years for simulation
        lookback_buffer = params.lookback_period + 5
        sim_len = min(756, len(close_prices) - lookback_buffer)
        sim_start = close_prices.index[-sim_len]

        sim_prices = close_prices.loc[sim_start:]
        sim_open = open_prices.loc[sim_start:]

        equity, exposures, weights = run_simulation(
            sim_prices,
            sim_open,
            close_prices,
            all_tickers,
            params,
            INITIAL_CAPITAL,
        )

        eq_series = pd.Series(equity, index=sim_prices.index[: len(equity)])
        metrics = compute_metrics(eq_series)
        print(f"\n{format_metrics_table(metrics)}")

        # Show top holdings
        weight_series = pd.Series(weights, index=all_tickers)
        top = weight_series[weight_series > 0].sort_values(ascending=False).head(10)
        if not top.empty:
            print("\nTop Holdings:")
            for t, w in top.items():
                print(f"  {t}: {w:.1%}")


if __name__ == "__main__":
    main()
