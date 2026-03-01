"""CLI command for walk-forward optimization (hybrid R\u00b2 Momentum + vol-targeting)."""

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
    run_schedule_optimization,
    run_walk_forward,
)

COMMAND_NAME = "walk-forward"


def register(subparsers) -> None:
    p = subparsers.add_parser(
        COMMAND_NAME,
        help="Walk-forward optimization (hybrid R\u00b2 Momentum + vol-targeting)",
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
        "--n-trials", type=int, default=100,
        help="Number of Optuna trials per WFO step (default: 100)",
    )
    p.add_argument(
        "--oos-days", type=int, default=None,
        help="OOS window size in trading days (default: 21)",
    )
    p.add_argument(
        "--min-is-days", type=int, default=None,
        help="IS window size in trading days (default: 126)",
    )
    p.add_argument(
        "--target-vol", type=float, default=None,
        help="Target annualised portfolio vol (default: 0.10 = 10%%)",
    )
    p.add_argument(
        "--max-leverage", type=float, default=None,
        help="Max scale factor for vol targeting (default: 1.5)",
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
        "--metric", default="total_return",
        choices=("total_return", "cagr", "sharpe", "calmar"),
        help="Optimization metric (default: total_return)",
    )
    p.add_argument(
        "--optimize-schedule", action="store_true",
        help="Optimize WFO schedule params via outer Optuna loop",
    )
    p.add_argument(
        "--n-schedule-trials", type=int, default=20,
        help="Number of outer Optuna trials for schedule optimization (default: 20)",
    )


def run(args) -> None:
    setup_logging()
    output_dir = create_output_dir("wfo")

    print("\nHybrid R\u00b2 Momentum + Vol-Targeting Strategy")
    print("=" * 50)
    print("Using cross-asset ETF universe...")
    tickers = fetch_etf_tickers()
    print(f"Universe: {len(tickers)} tickers")

    # Build base params with CLI overrides
    param_overrides = {}
    if args.target_vol is not None:
        param_overrides["target_vol"] = args.target_vol
    if args.max_leverage is not None:
        param_overrides["max_leverage"] = args.max_leverage

    base_params = StrategyParams(**param_overrides)

    # Resolve weeks -> days
    if args.oos_weeks is not None:
        oos_days = args.oos_weeks * 5
    else:
        oos_days = args.oos_days
    if args.min_is_weeks is not None:
        min_is_days = args.min_is_weeks * 5
    else:
        min_is_days = args.min_is_days

    min_days = base_params.r2_lookback
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

    if not args.optimize_schedule and oos_days is not None and min_is_days is not None and oos_days > min_is_days:
        print(f"\nERROR: OOS period ({oos_days}d) must be <= IS period ({min_is_days}d)")
        sys.exit(1)

    if args.optimize_schedule:
        print(f"\nStarting schedule optimization...")
        print(f"  Schedule trials: {args.n_schedule_trials}")
        print(f"  Inner trials per step: {args.n_trials}")
        print(f"  Target vol:   {base_params.target_vol:.0%}")
        print(f"  Max leverage:  {base_params.max_leverage:.2f}")
        print(f"  Metric: {args.metric}")
        print()

        result = run_schedule_optimization(
            close_prices,
            open_prices,
            valid_tickers,
            INITIAL_CAPITAL,
            base_params=base_params,
            n_trials_per_step=args.n_trials,
            n_schedule_trials=args.n_schedule_trials,
            n_workers=args.n_workers,
            metric=args.metric,
        )
    else:
        display_is = min_is_days or 126
        display_oos = oos_days or 21
        print(f"\nStarting walk-forward optimization...")
        print(f"  IS window:    {display_is} days (~{display_is / 21:.0f} months)")
        print(f"  OOS window:   {display_oos} days (~{display_oos / 21:.0f} weeks)")
        print(f"  Trials/step:  {args.n_trials}")
        print(f"  Target vol:   {base_params.target_vol:.0%}")
        print(f"  Max leverage:  {base_params.max_leverage:.2f}")
        print(f"  Metric: {args.metric}")
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
            metric=args.metric,
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
        title="Hybrid WFO: OOS Equity (Stitched)",
        filename="stitched_oos_equity.png",
    )

    step_rows = []
    for step in result.steps:
        p = step.optimized_params
        row = {
            "step": step.step_index + 1,
            "is_start": step.is_start.date(),
            "is_end": step.is_end.date(),
            "oos_start": step.oos_start.date(),
            "oos_end": step.oos_end.date(),
            "r2_lookback": p.r2_lookback,
            "kama_asset_period": p.kama_asset_period,
            "kama_spy_period": p.kama_spy_period,
            "kama_buffer": p.kama_buffer,
            "atr_period": p.atr_period,
            "top_n": p.top_n,
            "rebal_period_weeks": p.rebal_period_weeks,
            "gap_threshold": p.gap_threshold,
            "corr_threshold": p.corr_threshold,
            "target_vol": p.target_vol,
            "max_leverage": p.max_leverage,
            "portfolio_vol_lookback": p.portfolio_vol_lookback,
            "is_cagr": step.is_metrics.get("cagr", 0),
            "is_maxdd": step.is_metrics.get("max_drawdown", 0),
            "is_sharpe": step.is_metrics.get("sharpe", 0),
            "oos_cagr": step.oos_metrics.get("cagr", 0),
            "oos_maxdd": step.oos_metrics.get("max_drawdown", 0),
            "oos_sharpe": step.oos_metrics.get("sharpe", 0),
        }
        step_rows.append(row)

    steps_df = pd.DataFrame(step_rows)
    steps_path = output_dir / "wfo_steps.csv"
    steps_df.to_csv(steps_path, index=False)
    print(f"Step details saved to {steps_path}")

    fp = result.final_params
    print(f"\nRecommended live parameters:")
    print(f"  r2_lookback={fp.r2_lookback}, kama_asset={fp.kama_asset_period}, "
          f"kama_spy={fp.kama_spy_period}, kama_buffer={fp.kama_buffer}")
    print(f"  atr_period={fp.atr_period}, top_n={fp.top_n}, "
          f"rebal={fp.rebal_period_weeks}w, gap={fp.gap_threshold}")
    print(f"  target_vol={fp.target_vol}, max_leverage={fp.max_leverage}, "
          f"portfolio_vol_lookback={fp.portfolio_vol_lookback}")
    print(f"  corr_threshold={fp.corr_threshold}")
