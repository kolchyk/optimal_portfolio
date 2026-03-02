"""CLI command for parameter stability analysis via walk-forward optimization."""

import sys

from src.portfolio_sim.cli_utils import (
    create_output_dir,
    filter_valid_tickers,
    setup_logging,
)
from src.portfolio_sim.config import (
    DEFAULT_MIN_IS_DAYS,
    DEFAULT_OOS_DAYS,
    SEARCH_SPACE,
)
from src.portfolio_sim.data import fetch_etf_tickers, fetch_price_data
from src.portfolio_sim.params import StrategyParams
from src.portfolio_sim.stability import (
    format_stability_report,
    generate_candidate_values,
    run_param_stability,
    save_stability_charts,
    save_stability_csvs,
)

COMMAND_NAME = "param-stability"


def register(subparsers) -> None:
    p = subparsers.add_parser(
        COMMAND_NAME,
        help="Parameter stability analysis via walk-forward optimization",
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
        help="Number of parallel workers (default: cpu_count)",
    )
    p.add_argument(
        "--n-trials", type=int, default=50,
        help="Optuna trials per WFO step (default: 50, reduced for speed)",
    )
    p.add_argument(
        "--oos-days", type=int, default=None,
        help=f"OOS window size in trading days (default: {DEFAULT_OOS_DAYS})",
    )
    p.add_argument(
        "--min-is-days", type=int, default=None,
        help=f"IS window size in trading days (default: {DEFAULT_MIN_IS_DAYS})",
    )
    p.add_argument(
        "--oos-weeks", type=int, default=None,
        help="OOS window in weeks (overrides --oos-days; converted to days x 5)",
    )
    p.add_argument(
        "--min-is-weeks", type=int, default=None,
        help="IS window in weeks (overrides --min-is-days; converted to days x 5)",
    )
    p.add_argument(
        "--metric", default="sharpe",
        choices=("total_return", "cagr", "sharpe", "calmar"),
        help="Optimization & reporting metric (default: sharpe)",
    )
    p.add_argument(
        "--params", nargs="+", default=None, metavar="PARAM",
        help="Subset of parameter names to test (default: all). "
             f"Choices: {', '.join(SEARCH_SPACE.keys())}",
    )
    p.add_argument(
        "--max-points", type=int, default=None,
        help="Max candidate values per parameter (subsampled if exceeded)",
    )


def run(args) -> None:
    setup_logging()
    output_dir = create_output_dir("stability")

    print("\nParameter Stability Analysis (OAT via WFO)")
    print("=" * 50)

    # Validate --params
    param_names = args.params
    if param_names:
        unknown = set(param_names) - set(SEARCH_SPACE.keys())
        if unknown:
            print(f"\nERROR: Unknown parameter(s): {', '.join(sorted(unknown))}")
            print(f"Valid: {', '.join(SEARCH_SPACE.keys())}")
            sys.exit(1)

    # Load data
    tickers = fetch_etf_tickers()
    print(f"Universe: {len(tickers)} tickers")

    base_params = StrategyParams()
    min_days = base_params.r2_window

    print(f"Downloading price data ({args.period})...")
    close_prices, open_prices, high_prices, low_prices = fetch_price_data(
        tickers, period=args.period, refresh=args.refresh,
        cache_suffix="_etf", min_rows=min_days,
    )

    valid_tickers = filter_valid_tickers(close_prices, min_days)
    print(f"Tradable tickers with {min_days}+ days: {len(valid_tickers)}")

    if not valid_tickers:
        print(f"\nERROR: No tickers with {min_days}+ trading days.")
        sys.exit(1)

    # Resolve weeks -> days
    oos_days = args.oos_weeks * 5 if args.oos_weeks is not None else args.oos_days
    min_is_days = args.min_is_weeks * 5 if args.min_is_weeks is not None else args.min_is_days

    if oos_days is not None and min_is_days is not None and oos_days > min_is_days:
        print(f"\nERROR: OOS period ({oos_days}d) must be <= IS period ({min_is_days}d)")
        sys.exit(1)

    # Estimate run count
    test_params = param_names or list(SEARCH_SPACE.keys())
    total_runs = sum(
        len(generate_candidate_values(p, SEARCH_SPACE, args.max_points))
        for p in test_params
    )

    display_is = min_is_days or DEFAULT_MIN_IS_DAYS
    display_oos = oos_days or DEFAULT_OOS_DAYS
    print(f"\nStability analysis plan:")
    print(f"  Parameters:    {len(test_params)} ({', '.join(test_params)})")
    print(f"  Total WFO runs: {total_runs}")
    print(f"  Trials/step:   {args.n_trials}")
    print(f"  IS window:     {display_is} days (~{display_is / 21:.0f} months)")
    print(f"  OOS window:    {display_oos} days (~{display_oos / 21:.0f} weeks)")
    print(f"  Metric:        {args.metric}")
    print()

    result = run_param_stability(
        close_prices, open_prices, valid_tickers,
        base_params=base_params,
        param_names=param_names,
        n_trials_per_step=args.n_trials,
        n_workers=args.n_workers,
        min_is_days=min_is_days,
        oos_days=oos_days,
        metric=args.metric,
        max_points_per_param=args.max_points,
        high_prices=high_prices,
        low_prices=low_prices,
    )

    # Report
    report = format_stability_report(result, metric=args.metric)
    print(f"\n{report}")

    report_path = output_dir / "stability_report.txt"
    report_path.write_text(report)
    print(f"\nReport saved to {report_path}")

    # CSVs
    save_stability_csvs(result, output_dir)

    # Charts
    save_stability_charts(result, output_dir, metric=args.metric)

    print(f"\nAll artifacts saved to {output_dir}/")
