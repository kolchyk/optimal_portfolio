"""CLI entry point for parameter sensitivity analysis.

Usage:
    python run_optimizer.py
    python run_optimizer.py --period 10y
    python run_optimizer.py --period 10y --n-workers 4
    python run_optimizer.py --refresh              # Force re-download data
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import structlog

from src.portfolio_sim.config import INITIAL_CAPITAL
from src.portfolio_sim.data import fetch_price_data, fetch_sp500_tickers
from src.portfolio_sim.engine import run_simulation
from src.portfolio_sim.optimizer import (
    format_sensitivity_report,
    run_sensitivity,
)
from src.portfolio_sim.params import StrategyParams
from src.portfolio_sim.reporting import format_asset_report, save_equity_png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parameter sensitivity analysis for KAMA momentum strategy"
    )
    parser.add_argument(
        "--refresh", action="store_true",
        help="Force refresh data cache from yfinance",
    )
    parser.add_argument(
        "--period", default="10y",
        help="yfinance period string (default: 10y)",
    )
    parser.add_argument(
        "--n-workers", type=int, default=None,
        help="Number of parallel workers (default: cpu_count - 1)",
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
    output_dir = Path("output") / f"sens_{dt}"
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

    # Run sensitivity analysis
    base_params = StrategyParams()
    print(f"\nStarting sensitivity analysis...")
    print(f"  Base params: kama={base_params.kama_period}, "
          f"lookback={base_params.lookback_period}, "
          f"buffer={base_params.kama_buffer}, top_n={base_params.top_n}")
    print()

    result = run_sensitivity(
        close_prices,
        open_prices,
        valid_tickers,
        INITIAL_CAPITAL,
        base_params=base_params,
        n_workers=args.n_workers,
    )

    # Report
    report = format_sensitivity_report(result)
    print(f"\n{report}")

    # Save report
    report_path = output_dir / "sensitivity_report.txt"
    report_path.write_text(report)
    print(f"\nReport saved to {report_path}")

    # Save full grid results
    grid_path = output_dir / "grid_results.csv"
    result.grid_results.to_csv(grid_path, index=False)
    print(f"Grid results saved to {grid_path}")

    # Run base-params simulation for equity chart and asset report
    print("\nRunning base-params simulation for detailed report...")
    sim_result = run_simulation(
        close_prices, open_prices, valid_tickers, INITIAL_CAPITAL,
        params=base_params,
    )

    # Equity curve chart
    save_equity_png(sim_result.equity, sim_result.spy_equity, output_dir)

    # Asset report
    meta_path = Path("sp500_companies.csv")
    asset_meta = pd.read_csv(meta_path) if meta_path.exists() else None
    asset_report = format_asset_report(sim_result, close_prices, asset_meta)
    asset_report_path = output_dir / "asset_report.txt"
    asset_report_path.write_text(asset_report)
    print(f"Asset report saved to {asset_report_path}")


if __name__ == "__main__":
    main()
