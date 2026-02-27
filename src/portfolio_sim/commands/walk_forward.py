"""Walk-forward optimization."""

import sys

import pandas as pd

from src.portfolio_sim.cli_utils import (
    create_output_dir,
    filter_valid_tickers,
    setup_logging,
)
from src.portfolio_sim.config import INITIAL_CAPITAL
from src.portfolio_sim.data import fetch_etf_tickers, fetch_price_data
from src.portfolio_sim.params import StrategyParams
from src.portfolio_sim.reporting import save_equity_png
from src.portfolio_sim.walk_forward import (
    format_wfo_report,
    run_walk_forward,
)

COMMAND_NAME = "walk-forward"


def register(subparsers) -> None:
    p = subparsers.add_parser(
        COMMAND_NAME, help="Walk-forward optimization",
    )
    p.add_argument(
        "--refresh", action="store_true",
        help="Force refresh data cache from yfinance",
    )
    p.add_argument(
        "--period", default="3y",
        help="yfinance period string (default: 3y)",
    )
    p.add_argument(
        "--n-workers", type=int, default=None,
        help="Number of parallel workers (default: cpu_count - 1)",
    )
    p.add_argument(
        "--n-trials", type=int, default=150,
        help="Number of Optuna trials per WFO step (default: 150)",
    )
    p.add_argument(
        "--oos-days", type=int, default=None,
        help="OOS window size in trading days (default: 21 ≈ 1 month)",
    )
    p.add_argument(
        "--min-is-days", type=int, default=None,
        help="IS window size in trading days (default: 126 ≈ 6 months)",
    )


def run(args) -> None:
    setup_logging()
    output_dir = create_output_dir("wfo")

    print("\nUsing cross-asset ETF universe...")
    tickers = fetch_etf_tickers()
    print(f"Universe: {len(tickers)} tickers")

    base_params = StrategyParams()
    if args.oos_days is not None:
        base_params = StrategyParams(oos_days=args.oos_days)

    min_is_days = args.min_is_days  # None → run_walk_forward uses its default (126)
    oos_days = args.oos_days        # None → run_walk_forward uses its default (21)

    min_days = base_params.lookback_period
    print(f"Downloading price data ({args.period})...")
    close_prices, open_prices = fetch_price_data(
        tickers, period=args.period, refresh=args.refresh,
        cache_suffix="_etf", min_rows=min_days,
    )

    valid_tickers = filter_valid_tickers(close_prices, min_days)
    print(f"Tradable tickers with {min_days}+ days: {len(valid_tickers)}")

    if not valid_tickers:
        print(f"\nERROR: No tickers with {min_days}+ trading days.")
        print("Try: python -m src.portfolio_sim walk-forward --refresh")
        sys.exit(1)

    display_is = min_is_days or 126
    display_oos = oos_days or 21
    print(f"\nStarting walk-forward optimization...")
    print(f"  IS window: {display_is} days (~{display_is / 21:.0f} months)")
    print(f"  OOS window: {display_oos} days (~{display_oos / 21:.0f} weeks)")
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
        min_is_days=min_is_days,
        oos_days=oos_days,
    )

    report = format_wfo_report(result)
    print(f"\n{report}")

    report_path = output_dir / "wfo_report.txt"
    report_path.write_text(report)
    print(f"\nReport saved to {report_path}")

    _save_wfo_artifacts(result, output_dir)


def _save_wfo_artifacts(result, output_dir) -> None:
    """Save equity CSV, PNG, and step details for a WFO result."""
    equity_path = output_dir / "stitched_oos_equity.csv"
    result.stitched_equity.to_csv(equity_path, header=True)
    print(f"Stitched OOS equity saved to {equity_path}")

    save_equity_png(
        result.stitched_equity,
        result.stitched_spy_equity,
        output_dir,
        title="Walk-Forward OOS Equity (Stitched)",
        filename="stitched_oos_equity.png",
    )

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
            "oos_days": step.optimized_params.oos_days,
            "corr_threshold": step.optimized_params.corr_threshold,
            "is_cagr": step.is_metrics.get("cagr", 0),
            "is_maxdd": step.is_metrics.get("max_drawdown", 0),
            "oos_cagr": step.oos_metrics.get("cagr", 0),
            "oos_maxdd": step.oos_metrics.get("max_drawdown", 0),
            "oos_sharpe": step.oos_metrics.get("sharpe", 0),
        })

    steps_df = pd.DataFrame(step_rows)
    steps_path = output_dir / "wfo_steps.csv"
    steps_df.to_csv(steps_path, index=False)
    print(f"Step details saved to {steps_path}")

    fp = result.final_params
    print(f"\nRecommended live parameters:")
    print(f"  kama_period={fp.kama_period}, lookback_period={fp.lookback_period}, "
          f"kama_buffer={fp.kama_buffer}, top_n={fp.top_n}, "
          f"oos_days={fp.oos_days}, corr_threshold={fp.corr_threshold}")
