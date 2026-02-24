"""CLI entry point for walk-forward parameter optimization.

Usage:
    python run_optimizer.py
    python run_optimizer.py --period 10y
    python run_optimizer.py --period 10y --n-workers 4
    python run_optimizer.py --refresh              # Force re-download data
    python run_optimizer.py --min-is-years 3 --oos-years 1
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

import structlog

from src.portfolio_sim.config import INITIAL_CAPITAL
from src.portfolio_sim.data import fetch_price_data, fetch_sp500_tickers
from src.portfolio_sim.optimizer import (
    WalkForwardConfig,
    format_walk_forward_report,
    run_walk_forward,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Walk-forward parameter optimization for KAMA momentum strategy"
    )
    parser.add_argument(
        "--refresh", action="store_true",
        help="Force refresh data cache from yfinance",
    )
    parser.add_argument(
        "--period", default="10y",
        help="yfinance period string (default: 10y for more folds)",
    )
    parser.add_argument(
        "--n-workers", type=int, default=None,
        help="Number of parallel workers (default: cpu_count - 1)",
    )
    parser.add_argument(
        "--min-is-years", type=int, default=3,
        help="Minimum in-sample window in years (default: 3)",
    )
    parser.add_argument(
        "--oos-years", type=int, default=1,
        help="Out-of-sample window in years (default: 1)",
    )
    parser.add_argument(
        "--max-dd", type=float, default=0.30,
        help="Max drawdown limit for parameter rejection (default: 0.30)",
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
    output_dir = Path("output") / f"opt_{dt}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fetch data
    print("\nFetching S&P 500 constituents...")
    sp500_tickers = fetch_sp500_tickers()
    print(f"S&P 500: {len(sp500_tickers)} tickers")

    print(f"Downloading price data ({args.period})...")
    close_prices, open_prices = fetch_price_data(
        sp500_tickers, period=args.period, refresh=args.refresh
    )

    # Filter tickers with sufficient history
    min_days = 756
    valid_tickers = [
        t for t in close_prices.columns
        if t != "SPY" and len(close_prices[t].dropna()) >= min_days
    ]
    print(f"Tradable tickers with {min_days}+ days: {len(valid_tickers)}")

    # Walk-forward config
    wf_config = WalkForwardConfig(
        min_is_days=args.min_is_years * 252,
        oos_days=args.oos_years * 252,
        step_days=args.oos_years * 252,
        max_drawdown_limit=args.max_dd,
    )

    # Run optimization
    print(f"\nStarting walk-forward optimization...")
    print(f"  IS window:  >= {args.min_is_years} years (expanding)")
    print(f"  OOS window: {args.oos_years} year(s)")
    print(f"  Grid size:  625 parameter combinations")
    print()

    result = run_walk_forward(
        close_prices,
        open_prices,
        valid_tickers,
        INITIAL_CAPITAL,
        wf_config=wf_config,
        n_workers=args.n_workers,
    )

    # Report
    report = format_walk_forward_report(result)
    print(f"\n{report}")

    # Save report
    report_path = output_dir / "walk_forward_report.txt"
    report_path.write_text(report)
    print(f"\nReport saved to {report_path}")

    # Save stability table
    stability_path = output_dir / "parameter_stability.csv"
    result.parameter_stability.to_csv(stability_path, index=False)
    print(f"Stability table saved to {stability_path}")


if __name__ == "__main__":
    main()
