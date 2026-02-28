"""CLI command for V2 walk-forward optimization (vol-targeted)."""

import sys

import pandas as pd

from src.portfolio_sim.cli_utils import (
    create_output_dir,
    filter_valid_tickers,
    setup_logging,
)
from src.portfolio_sim.config import INITIAL_CAPITAL
from src.portfolio_sim.data import fetch_etf_tickers, fetch_price_data
from src.portfolio_sim.reporting import save_equity_png
from src.portfolio_sim.strategy_v2.params import StrategyParamsV2
from src.portfolio_sim.strategy_v2.walk_forward import (
    format_wfo_report_v2,
    run_walk_forward_v2,
)

COMMAND_NAME = "walk-forward-v2"


def register(subparsers) -> None:
    p = subparsers.add_parser(
        COMMAND_NAME, help="Walk-forward optimization (V2: vol-targeted, Sharpe objective)",
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


def run(args) -> None:
    setup_logging()
    output_dir = create_output_dir("wfo_v2")

    print("\n[V2] Vol-Targeted KAMA Momentum Strategy")
    print("=" * 50)
    print("Using cross-asset ETF universe...")
    tickers = fetch_etf_tickers()
    print(f"Universe: {len(tickers)} tickers")

    # Build base params with CLI overrides
    param_overrides = {}
    if args.oos_days is not None:
        param_overrides["oos_days"] = args.oos_days
    if args.target_vol is not None:
        param_overrides["target_vol"] = args.target_vol
    if args.max_leverage is not None:
        param_overrides["max_leverage"] = args.max_leverage

    base_params = StrategyParamsV2(**param_overrides)

    min_is_days = args.min_is_days
    oos_days = args.oos_days

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
        print("Try: python -m src.portfolio_sim walk-forward-v2 --refresh")
        sys.exit(1)

    display_is = min_is_days or 126
    display_oos = oos_days or 21
    print(f"\nStarting V2 walk-forward optimization...")
    print(f"  IS window:    {display_is} days (~{display_is / 21:.0f} months)")
    print(f"  OOS window:   {display_oos} days (~{display_oos / 21:.0f} weeks)")
    print(f"  Trials/step:  {args.n_trials}")
    print(f"  Target vol:   {base_params.target_vol:.0%}")
    print(f"  Max leverage:  {base_params.max_leverage:.2f}")
    print(f"  Objective:     Sharpe ratio (max DD <= 20%)")
    print()

    result = run_walk_forward_v2(
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

    report = format_wfo_report_v2(result)
    print(f"\n{report}")

    report_path = output_dir / "wfo_v2_report.txt"
    report_path.write_text(report)
    print(f"\nReport saved to {report_path}")

    _save_wfo_v2_artifacts(result, output_dir)


def _save_wfo_v2_artifacts(result, output_dir) -> None:
    """Save equity CSV, PNG, and step details for a V2 WFO result."""
    equity_path = output_dir / "stitched_oos_equity.csv"
    result.stitched_equity.to_csv(equity_path, header=True)
    print(f"Stitched OOS equity saved to {equity_path}")

    save_equity_png(
        result.stitched_equity,
        result.stitched_spy_equity,
        output_dir,
        title="V2 Vol-Targeted WFO: OOS Equity (Stitched)",
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
            "kama_period": p.kama_period,
            "lookback_period": p.lookback_period,
            "kama_buffer": p.kama_buffer,
            "top_n": p.top_n,
            "oos_days": p.oos_days,
            "corr_threshold": p.corr_threshold,
            "weighting_mode": p.weighting_mode,
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
    steps_path = output_dir / "wfo_v2_steps.csv"
    steps_df.to_csv(steps_path, index=False)
    print(f"Step details saved to {steps_path}")

    fp = result.final_params
    print(f"\nRecommended live parameters (V2):")
    print(f"  kama_period={fp.kama_period}, lookback_period={fp.lookback_period}, "
          f"kama_buffer={fp.kama_buffer}, top_n={fp.top_n}")
    print(f"  target_vol={fp.target_vol}, max_leverage={fp.max_leverage}, "
          f"portfolio_vol_lookback={fp.portfolio_vol_lookback}")
    print(f"  corr_threshold={fp.corr_threshold}, weighting_mode={fp.weighting_mode}")
