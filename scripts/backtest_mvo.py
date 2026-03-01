"""MVO benchmark backtest with configurable parameters.

Usage:
    uv run python scripts/backtest_mvo.py
    uv run python scripts/backtest_mvo.py --period 5y
    uv run python scripts/backtest_mvo.py --objective max_sharpe
    uv run python scripts/backtest_mvo.py --objective risk_parity --rebal-freq quarter
"""

from __future__ import annotations

import argparse

from src.portfolio_sim.benchmark_mvo.engine import run_mvo_backtest
from src.portfolio_sim.benchmark_mvo.params import MVOParams
from src.portfolio_sim.cli_utils import create_output_dir, filter_valid_tickers, setup_logging
from src.portfolio_sim.config import INITIAL_CAPITAL, SPY_TICKER
from src.portfolio_sim.data import fetch_etf_tickers, fetch_price_data
from src.portfolio_sim.reporting import compute_metrics, format_comparison_table, save_equity_png


def main() -> None:
    parser = argparse.ArgumentParser(description="MVO benchmark backtest")
    parser.add_argument("--period", default="3y", help="yfinance period (default: 3y)")
    parser.add_argument("--refresh", action="store_true", help="Force refresh data cache")
    parser.add_argument(
        "--objective", default="min_variance",
        choices=("min_variance", "max_sharpe", "risk_parity"),
        help="MVO objective (default: min_variance)",
    )
    parser.add_argument("--cov-lookback", type=int, default=252, help="Covariance lookback days")
    parser.add_argument("--max-weight", type=float, default=0.20, help="Max weight per asset")
    parser.add_argument(
        "--rebal-freq", default="month", choices=("month", "quarter"),
        help="Rebalance frequency",
    )
    args = parser.parse_args()

    setup_logging()

    params = MVOParams(
        objective=args.objective,
        cov_lookback=args.cov_lookback,
        max_weight=args.max_weight,
        rebal_freq=args.rebal_freq,
    )

    obj_label = args.objective.replace("_", " ").title()
    print("=" * 60)
    print(f"MVO Benchmark Backtest — {obj_label}")
    print("=" * 60)
    print(f"\nParams: cov_lb={params.cov_lookback} max_w={params.max_weight:.0%} "
          f"rebal={params.rebal_freq} objective={params.objective}")

    # Load data
    print("\nLoading data...")
    tickers = fetch_etf_tickers()
    close_prices, open_prices = fetch_price_data(
        tickers, period=args.period, refresh=args.refresh, cache_suffix="_etf",
    )

    min_days = params.warmup
    valid_tickers = [
        t for t in filter_valid_tickers(close_prices, min_days)
        if t != SPY_TICKER
    ]
    print(f"  Universe: {len(valid_tickers)} tickers, "
          f"{close_prices.index[0].date()} .. {close_prices.index[-1].date()}")

    if not valid_tickers:
        print("ERROR: No valid tickers.")
        return

    # Run backtest
    print("\nRunning backtest...")
    equity, spy_equity, _, _, trade_log = run_mvo_backtest(
        close_prices, open_prices, valid_tickers,
        initial_capital=INITIAL_CAPITAL, params=params,
    )

    if equity.empty:
        print("ERROR: Empty equity curve.")
        return

    # Report
    strat_metrics = compute_metrics(equity)
    spy_metrics = compute_metrics(spy_equity)
    print("\n" + format_comparison_table(strat_metrics, spy_metrics))
    print(f"\n  Trades: {len(trade_log)}")

    # Save
    output_dir = create_output_dir("backtest_mvo")
    print(f"\nSaving to {output_dir}/")

    (output_dir / "equity.csv").write_text(equity.to_csv(), encoding="utf-8")
    save_equity_png(
        equity, spy_equity, output_dir,
        title=f"MVO {obj_label} Benchmark",
        filename="equity_curve.png",
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
