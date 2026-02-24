"""CLI entry point for KAMA momentum portfolio simulation.

Usage:
    python run_portfolio_sim.py
    python run_portfolio_sim.py --refresh
    python run_portfolio_sim.py --period 10y
    python run_portfolio_sim.py --cache-only --refresh   # Download & save to cache, skip sim
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

import structlog

from src.portfolio_sim.config import INITIAL_CAPITAL
from src.portfolio_sim.data import fetch_price_data, fetch_sp500_tickers
from src.portfolio_sim.engine import run_simulation
from src.portfolio_sim.reporting import (
    compute_metrics,
    format_comparison_table,
    save_equity_png,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KAMA Momentum Strategy -- S&P 500 backtest"
    )
    parser.add_argument(
        "--refresh", action="store_true",
        help="Force refresh data cache from yfinance",
    )
    parser.add_argument(
        "--period", default="5y",
        help="yfinance period string (default: 5y)",
    )
    parser.add_argument(
        "--cache-only", action="store_true",
        help="Only fetch data and save to cache; skip simulation. Use with --refresh to populate cache.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

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
    dt = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output") / f"sim_{dt}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fetch S&P 500 tickers and price data
    print("\nFetching S&P 500 constituents...")
    sp500_tickers = fetch_sp500_tickers()
    print(f"S&P 500: {len(sp500_tickers)} tickers")

    if args.cache_only:
        print(f"Downloading price data ({args.period}) and saving to cache...")
    else:
        print(f"Downloading price data ({args.period})...")
    close_prices, open_prices = fetch_price_data(
        sp500_tickers, period=args.period, refresh=args.refresh or args.cache_only
    )

    if args.cache_only:
        print(f"Cache saved. {len(close_prices.columns)} tickers in output/cache/.")
        print("Future runs will load from cache (no download) unless you use --refresh.")
        return

    # Filter tickers with sufficient history
    min_days = 756  # ~3 years
    valid_tickers = [
        t for t in close_prices.columns
        if t != "SPY" and len(close_prices[t].dropna()) >= min_days
    ]
    print(f"Tradable tickers with {min_days}+ days: {len(valid_tickers)}")

    # Run simulation
    print("\nRunning simulation...")
    result = run_simulation(
        close_prices, open_prices, valid_tickers, INITIAL_CAPITAL
    )

    # Report
    strat_metrics = compute_metrics(result.equity)
    spy_metrics = compute_metrics(result.spy_equity)

    print(f"\n{format_comparison_table(strat_metrics, spy_metrics)}")

    save_equity_png(result.equity, result.spy_equity, output_dir)
    print(f"\nOutput saved to {output_dir}")


if __name__ == "__main__":
    main()
