"""Parameter sensitivity analysis (Optuna TPE)."""

import sys

from src.portfolio_sim.cli_utils import (
    create_output_dir,
    filter_valid_tickers,
    setup_logging,
)
from src.portfolio_sim.config import INITIAL_CAPITAL
from src.portfolio_sim.data import fetch_etf_tickers, fetch_price_data
from src.portfolio_sim.engine import run_simulation
from src.portfolio_sim.optimizer import (
    format_sensitivity_report,
    run_sensitivity,
)
from src.portfolio_sim.params import StrategyParams
from src.portfolio_sim.reporting import format_asset_report, save_equity_png

COMMAND_NAME = "optimize"


def register(subparsers) -> None:
    p = subparsers.add_parser(
        COMMAND_NAME, help="Parameter sensitivity analysis (Optuna TPE)",
    )
    p.add_argument(
        "--refresh", action="store_true",
        help="Force refresh data cache from yfinance",
    )
    p.add_argument(
        "--period", default="10y",
        help="yfinance period string (default: 10y)",
    )
    p.add_argument(
        "--n-workers", type=int, default=None,
        help="Number of parallel workers (default: cpu_count - 1)",
    )
    p.add_argument(
        "--n-trials", type=int, default=200,
        help="Number of Optuna trials for sensitivity analysis (default: 200)",
    )


def run(args) -> None:
    setup_logging()
    output_dir = create_output_dir("sens")

    print("\nUsing cross-asset ETF universe...")
    tickers = fetch_etf_tickers()
    print(f"Universe: {len(tickers)} tickers")

    min_days = 756
    print(f"Downloading price data ({args.period})...")
    close_prices, open_prices = fetch_price_data(
        tickers, period=args.period, refresh=args.refresh,
        cache_suffix="_etf", min_rows=min_days,
    )

    valid_tickers = filter_valid_tickers(close_prices, min_days)
    print(f"Tradable tickers with {min_days}+ days: {len(valid_tickers)}")

    if not valid_tickers:
        print(f"\nERROR: No tickers with {min_days}+ trading days.")
        print("Try: python -m src.portfolio_sim optimize --refresh")
        sys.exit(1)

    base_params = StrategyParams()
    print(f"\nStarting sensitivity analysis (Optuna TPE, {args.n_trials} trials)...")
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
        n_trials=args.n_trials,
        n_workers=args.n_workers,
    )

    report = format_sensitivity_report(result)
    print(f"\n{report}")

    report_path = output_dir / "sensitivity_report.txt"
    report_path.write_text(report)
    print(f"\nReport saved to {report_path}")

    grid_path = output_dir / "trial_results.csv"
    result.grid_results.to_csv(grid_path, index=False)
    print(f"Trial results saved to {grid_path}")

    print("\nRunning base-params simulation for detailed report...")
    sim_result = run_simulation(
        close_prices, open_prices, valid_tickers, INITIAL_CAPITAL,
        params=base_params,
    )

    save_equity_png(sim_result.equity, sim_result.spy_equity, output_dir)

    asset_report = format_asset_report(sim_result, close_prices, asset_meta=None)
    asset_report_path = output_dir / "asset_report.txt"
    asset_report_path.write_text(asset_report)
    print(f"Asset report saved to {asset_report_path}")
