"""Maximum profit parameter search (TPE or Pareto NSGA-II)."""

from src.portfolio_sim.cli_utils import (
    create_output_dir,
    filter_valid_tickers,
    setup_logging,
)
from src.portfolio_sim.config import INITIAL_CAPITAL
from src.portfolio_sim.data import fetch_etf_tickers, fetch_price_data
from src.portfolio_sim.engine import run_simulation
from src.portfolio_sim.max_profit import (
    format_max_profit_report,
    format_pareto_report,
    run_max_profit_pareto,
    run_max_profit_search,
    select_best_from_pareto,
)
from src.portfolio_sim.params import StrategyParams
from src.portfolio_sim.reporting import compute_metrics, format_comparison_table

COMMAND_NAME = "max-profit"


def register(subparsers) -> None:
    p = subparsers.add_parser(
        COMMAND_NAME, help="Maximum profit parameter search (TPE or Pareto NSGA-II)",
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
        "--max-dd", type=float, default=0.60,
        help="Max drawdown rejection limit (default: 0.60 = 60%%)",
    )
    p.add_argument(
        "--n-trials", type=int, default=10,
        help="Number of Optuna trials per universe (default: 500)",
    )
    p.add_argument(
        "--pareto", action="store_true",
        help="Use multi-objective Pareto search (NSGA-II) instead of single-objective TPE",
    )


def _run_verification(close_prices, open_prices, tickers, initial_capital, params, universe_name):
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


def run(args) -> None:
    setup_logging()
    output_dir = create_output_dir("max_profit")

    summary_lines: list[str] = []
    summary_lines.append("=" * 90)
    summary_lines.append("MAXIMUM PROFIT SEARCH — SUMMARY")
    summary_lines.append(f"Period: {args.period}  |  Max DD limit: {args.max_dd:.0%}")
    summary_lines.append("=" * 90)

    # ETF
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
    valid_etf = filter_valid_tickers(close_etf, min_days)
    print(f"Tradable tickers with {min_days}+ days: {len(valid_etf)}")

    _run_verification(
        close_etf, open_etf, valid_etf, INITIAL_CAPITAL,
        etf_params, "Cross-Asset ETF",
    )

    fixed = {
        "enable_correlation_filter": True,
        "correlation_threshold": 0.65,
        "correlation_lookback": 60,
        "use_risk_adjusted": True,
        "sizing_mode": "risk_parity",
    }

    if args.pareto:
        print(f"\n{'=' * 70}")
        print(f"PARETO SEARCH (NSGA-II) — Cross-Asset ETF ({args.n_trials} trials)")
        print(f"{'=' * 70}\n")

        etf_result = run_max_profit_pareto(
            close_etf, open_etf, valid_etf, INITIAL_CAPITAL,
            universe="etf",
            default_params=etf_params,
            fixed_params=fixed,
            n_trials=args.n_trials,
            n_workers=args.n_workers,
        )

        report_etf = format_pareto_report(etf_result)
        best_pareto = select_best_from_pareto(etf_result)
        if best_pareto:
            print(f"\nBest from Pareto front (by Calmar):")
            print(f"  kama={best_pareto.kama_period}, lookback={best_pareto.lookback_period}, "
                  f"buffer={best_pareto.kama_buffer}, top_n={best_pareto.top_n}")

        if etf_result.pareto_front is not None:
            pareto_path = output_dir / "pareto_front_etf.csv"
            etf_result.pareto_front.to_csv(pareto_path, index=False)
            print(f"Pareto front saved to {pareto_path}")
    else:
        print(f"\n{'=' * 70}")
        print(f"OPTUNA SEARCH — Cross-Asset ETF ({args.n_trials} trials)")
        print(f"{'=' * 70}\n")

        etf_result = run_max_profit_search(
            close_etf, open_etf, valid_etf, INITIAL_CAPITAL,
            universe="etf",
            default_params=etf_params,
            fixed_params=fixed,
            n_trials=args.n_trials,
            n_workers=args.n_workers,
            max_dd_limit=args.max_dd,
        )

        report_etf = format_max_profit_report(etf_result)

    print(f"\n{report_etf}")

    etf_result.grid_results.to_csv(
        output_dir / "grid_results_etf.csv", index=False,
    )
    (output_dir / "report_etf.txt").write_text(report_etf)
    summary_lines.append("")
    summary_lines.append(report_etf)

    summary = "\n".join(summary_lines)
    (output_dir / "summary.txt").write_text(summary)
    print(f"\nAll results saved to {output_dir}")
