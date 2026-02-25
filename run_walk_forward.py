"""CLI entry point for walk-forward optimization.

Usage:
    python run_walk_forward.py
    python run_walk_forward.py --period 10y
    python run_walk_forward.py --period 10y --n-trials 100 --oos-days 252
    python run_walk_forward.py --refresh
"""

import argparse

from src.portfolio_sim.cli_utils import (
    create_output_dir,
    filter_valid_tickers,
    setup_logging,
)
from src.portfolio_sim.config import INITIAL_CAPITAL
from src.portfolio_sim.data import fetch_etf_tickers, fetch_price_data
from src.portfolio_sim.params import StrategyParams
from src.portfolio_sim.walk_forward import format_wfo_report, run_walk_forward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Walk-forward optimization for KAMA momentum strategy"
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
    parser.add_argument(
        "--n-trials", type=int, default=100,
        help="Number of Optuna trials per WFO step (default: 100)",
    )
    parser.add_argument(
        "--oos-days", type=int, default=252,
        help="OOS window size in trading days (default: 252 ~ 1 year)",
    )
    parser.add_argument(
        "--min-is-days", type=int, default=756,
        help="Minimum IS window size in trading days (default: 756 ~ 3 years)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    setup_logging()
    output_dir = create_output_dir("wfo")

    # Fetch data
    print("\nUsing cross-asset ETF universe...")
    tickers = fetch_etf_tickers()
    print(f"Universe: {len(tickers)} tickers")

    print(f"Downloading price data ({args.period})...")
    close_prices, open_prices = fetch_price_data(
        tickers, period=args.period, refresh=args.refresh, cache_suffix="_etf",
    )

    # Filter tickers with sufficient history
    min_days = args.min_is_days
    valid_tickers = filter_valid_tickers(close_prices, min_days)
    print(f"Tradable tickers with {min_days}+ days: {len(valid_tickers)}")

    # Run walk-forward optimization
    base_params = StrategyParams()
    print(f"\nStarting walk-forward optimization...")
    print(f"  IS minimum: {args.min_is_days} days (~{args.min_is_days / 252:.0f} years)")
    print(f"  OOS window: {args.oos_days} days (~{args.oos_days / 252:.0f} years)")
    print(f"  Trials per step: {args.n_trials}")
    print()

    result = run_walk_forward(
        close_prices,
        open_prices,
        valid_tickers,
        INITIAL_CAPITAL,
        base_params=base_params,
        n_trials_per_step=args.n_trials,
        n_workers=args.n_workers,
        min_is_days=args.min_is_days,
        oos_days=args.oos_days,
    )

    # Report
    report = format_wfo_report(result)
    print(f"\n{report}")

    # Save report
    report_path = output_dir / "wfo_report.txt"
    report_path.write_text(report)
    print(f"\nReport saved to {report_path}")

    # Save stitched OOS equity
    equity_path = output_dir / "stitched_oos_equity.csv"
    result.stitched_equity.to_csv(equity_path, header=True)
    print(f"Stitched OOS equity saved to {equity_path}")

    # Save per-step details
    step_rows = []
    for step in result.steps:
        step_rows.append({
            "step": step.step_index + 1,
            "is_start": step.is_start.date(),
            "is_end": step.is_end.date(),
            "oos_start": step.oos_start.date(),
            "oos_end": step.oos_end.date(),
            "kama_period": step.optimized_params.kama_period,
            "lookback_period": step.optimized_params.lookback_period,
            "kama_buffer": step.optimized_params.kama_buffer,
            "top_n": step.optimized_params.top_n,
            "is_cagr": step.is_metrics.get("cagr", 0),
            "is_maxdd": step.is_metrics.get("max_drawdown", 0),
            "oos_cagr": step.oos_metrics.get("cagr", 0),
            "oos_maxdd": step.oos_metrics.get("max_drawdown", 0),
            "oos_sharpe": step.oos_metrics.get("sharpe", 0),
        })

    import pandas as pd
    steps_df = pd.DataFrame(step_rows)
    steps_path = output_dir / "wfo_steps.csv"
    steps_df.to_csv(steps_path, index=False)
    print(f"Step details saved to {steps_path}")

    # Print recommended parameters
    fp = result.final_params
    print(f"\nRecommended live parameters:")
    print(f"  kama_period={fp.kama_period}, lookback_period={fp.lookback_period}, "
          f"kama_buffer={fp.kama_buffer}, top_n={fp.top_n}")


if __name__ == "__main__":
    main()
