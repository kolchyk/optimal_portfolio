"""CLI entry point for maximum profit parameter search.

Usage:
    python run_max_profit.py                          # Both universes, 5y
    python run_max_profit.py --universe sp500          # S&P 500 only
    python run_max_profit.py --universe etf             # ETF only
    python run_max_profit.py --period 5y --n-workers 8
    python run_max_profit.py --refresh                  # Force refresh cache
    python run_max_profit.py --max-dd 0.40              # Tighter drawdown limit
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

import structlog

from src.portfolio_sim.config import INITIAL_CAPITAL
from src.portfolio_sim.data import fetch_etf_tickers, fetch_price_data, fetch_sp500_tickers
from src.portfolio_sim.engine import run_simulation
from src.portfolio_sim.max_profit import (
    format_max_profit_report,
    run_max_profit_search,
)
from src.portfolio_sim.params import StrategyParams
from src.portfolio_sim.reporting import compute_metrics, format_comparison_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Maximum profit parameter search for KAMA momentum strategy"
    )
    parser.add_argument(
        "--universe", choices=["sp500", "etf", "both"], default="both",
        help="Which universe(s) to search (default: both)",
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
        "--n-workers", type=int, default=None,
        help="Number of parallel workers (default: cpu_count - 1)",
    )
    parser.add_argument(
        "--max-dd", type=float, default=0.60,
        help="Max drawdown rejection limit (default: 0.60 = 60%%)",
    )
    parser.add_argument(
        "--n-trials", type=int, default=500,
        help="Number of Optuna trials per universe (default: 500)",
    )
    return parser.parse_args()


def run_verification(
    close_prices, open_prices, tickers, initial_capital, params, universe_name,
):
    """Run simulation with given params and print metrics."""
    print(f"\n{'=' * 70}")
    print(f"VERIFICATION — {universe_name} (default params)")
    print(f"{'=' * 70}")

    result = run_simulation(
        close_prices, open_prices, tickers, initial_capital,
        params=params, show_progress=True,
    )

    strat_metrics = compute_metrics(result.equity)
    spy_metrics = compute_metrics(result.spy_equity)

    print(f"\n{format_comparison_table(strat_metrics, spy_metrics)}")
    return strat_metrics


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
    output_dir = Path("output") / f"max_profit_{dt}"
    output_dir.mkdir(parents=True, exist_ok=True)

    do_sp500 = args.universe in ("sp500", "both")
    do_etf = args.universe in ("etf", "both")

    summary_lines: list[str] = []
    summary_lines.append("=" * 90)
    summary_lines.append("MAXIMUM PROFIT SEARCH — SUMMARY")
    summary_lines.append(f"Period: {args.period}  |  Max DD limit: {args.max_dd:.0%}")
    summary_lines.append("=" * 90)

    # ------------------------------------------------------------------
    # S&P 500
    # ------------------------------------------------------------------
    if do_sp500:
        print("\nFetching S&P 500 constituents...")
        sp500_tickers = fetch_sp500_tickers()
        print(f"Universe: {len(sp500_tickers)} tickers")

        print(f"Downloading price data ({args.period})...")
        close_sp500, open_sp500 = fetch_price_data(
            sp500_tickers, period=args.period, refresh=args.refresh,
            cache_suffix="",
        )

        sp500_params = StrategyParams()
        min_days = sp500_params.warmup * 2
        valid_sp500 = [
            t for t in close_sp500.columns
            if t != "SPY" and len(close_sp500[t].dropna()) >= min_days
        ]
        print(f"Tradable tickers with {min_days}+ days: {len(valid_sp500)}")

        # Part 1: Verification
        run_verification(
            close_sp500, open_sp500, valid_sp500, INITIAL_CAPITAL,
            sp500_params, "S&P 500",
        )

        # Part 2: Optuna search
        print(f"\n{'=' * 70}")
        print(f"OPTUNA SEARCH — S&P 500 ({args.n_trials} trials)")
        print(f"{'=' * 70}\n")

        sp500_result = run_max_profit_search(
            close_sp500, open_sp500, valid_sp500, INITIAL_CAPITAL,
            universe="sp500",
            default_params=sp500_params,
            fixed_params={"enable_correlation_filter": False},
            n_trials=args.n_trials,
            n_workers=args.n_workers,
            max_dd_limit=args.max_dd,
        )

        report_sp500 = format_max_profit_report(sp500_result)
        print(f"\n{report_sp500}")

        # Save
        sp500_result.grid_results.to_csv(
            output_dir / "grid_results_sp500.csv", index=False,
        )
        (output_dir / "report_sp500.txt").write_text(report_sp500)
        summary_lines.append("")
        summary_lines.append(report_sp500)

    # ------------------------------------------------------------------
    # ETF
    # ------------------------------------------------------------------
    if do_etf:
        print("\nFetching ETF universe...")
        etf_tickers = fetch_etf_tickers()
        print(f"Universe: {len(etf_tickers)} tickers")

        print(f"Downloading price data ({args.period})...")
        close_etf, open_etf = fetch_price_data(
            etf_tickers, period=args.period, refresh=args.refresh,
            cache_suffix="_etf",
        )

        etf_params = StrategyParams(
            use_risk_adjusted=True,
            enable_regime_filter=False,
            enable_correlation_filter=True,
            sizing_mode="risk_parity",
        )
        min_days = etf_params.warmup * 2
        valid_etf = [
            t for t in close_etf.columns
            if len(close_etf[t].dropna()) >= min_days
        ]
        print(f"Tradable tickers with {min_days}+ days: {len(valid_etf)}")

        # Part 1: Verification
        run_verification(
            close_etf, open_etf, valid_etf, INITIAL_CAPITAL,
            etf_params, "Cross-Asset ETF",
        )

        # Part 2: Optuna search
        print(f"\n{'=' * 70}")
        print(f"OPTUNA SEARCH — Cross-Asset ETF ({args.n_trials} trials)")
        print(f"{'=' * 70}\n")

        etf_result = run_max_profit_search(
            close_etf, open_etf, valid_etf, INITIAL_CAPITAL,
            universe="etf",
            default_params=etf_params,
            fixed_params={
                "enable_correlation_filter": True,
                "correlation_threshold": 0.65,
                "correlation_lookback": 60,
            },
            n_trials=args.n_trials,
            n_workers=args.n_workers,
            max_dd_limit=args.max_dd,
        )

        report_etf = format_max_profit_report(etf_result)
        print(f"\n{report_etf}")

        # Save
        etf_result.grid_results.to_csv(
            output_dir / "grid_results_etf.csv", index=False,
        )
        (output_dir / "report_etf.txt").write_text(report_etf)
        summary_lines.append("")
        summary_lines.append(report_etf)

    # ------------------------------------------------------------------
    # Save combined summary
    # ------------------------------------------------------------------
    summary = "\n".join(summary_lines)
    (output_dir / "summary.txt").write_text(summary)
    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()
