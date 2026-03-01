"""Hybrid backtest with fixed parameters.

Runs hybrid R² Momentum + Vol-Targeting with default or WFO-recommended params.

Usage:
    uv run python scripts/backtest_s2.py
    uv run python scripts/backtest_s2.py --period 5y
"""

from __future__ import annotations

import argparse

from src.portfolio_sim.cli_utils import create_output_dir, filter_valid_tickers, setup_logging
from src.portfolio_sim.config import INITIAL_CAPITAL, SPY_TICKER
from src.portfolio_sim.data import fetch_etf_tickers, fetch_price_data
from src.portfolio_sim.engine import run_simulation
from src.portfolio_sim.params import StrategyParams
from src.portfolio_sim.reporting import compute_metrics, format_comparison_table, save_equity_png

# Default parameters (can be overridden by WFO output)
PARAMS = StrategyParams(
    r2_lookback=90,
    kama_asset_period=10,
    kama_spy_period=40,
    kama_buffer=0.005,
    atr_period=20,
    risk_factor=0.001,
    top_n=5,
    rebal_period_weeks=3,
    gap_threshold=0.175,
    corr_threshold=0.7,
    target_vol=0.10,
    max_leverage=1.5,
    portfolio_vol_lookback=21,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hybrid backtest (R\u00b2 Momentum + Vol-Targeting)",
    )
    parser.add_argument("--period", default="3y", help="yfinance period (default: 3y)")
    parser.add_argument("--refresh", action="store_true", help="Force refresh data cache")
    args = parser.parse_args()

    setup_logging()

    print("=" * 60)
    print("Hybrid Backtest \u2014 R\u00b2 Momentum + Vol-Targeting")
    print("=" * 60)
    print(f"\nParams: r2_lb={PARAMS.r2_lookback} kama_asset={PARAMS.kama_asset_period} "
          f"kama_spy={PARAMS.kama_spy_period} kama_buf={PARAMS.kama_buffer} "
          f"gap={PARAMS.gap_threshold} atr={PARAMS.atr_period} top_n={PARAMS.top_n} "
          f"rebal={PARAMS.rebal_period_weeks}w")
    print(f"  target_vol={PARAMS.target_vol:.0%} max_leverage={PARAMS.max_leverage} "
          f"vol_lookback={PARAMS.portfolio_vol_lookback}")

    # 1. Load data
    print("\nLoading data...")
    tickers = fetch_etf_tickers()
    close_prices, open_prices = fetch_price_data(
        tickers, period=args.period, refresh=args.refresh, cache_suffix="_etf",
    )

    min_days = PARAMS.warmup
    valid_tickers = [
        t for t in filter_valid_tickers(close_prices, min_days)
        if t != SPY_TICKER
    ]
    print(f"  Universe: {len(valid_tickers)} tickers, "
          f"{close_prices.index[0].date()} .. {close_prices.index[-1].date()}")

    if not valid_tickers:
        print("ERROR: No valid tickers.")
        return

    # 2. Run backtest
    print("\nRunning backtest...")
    result = run_simulation(
        close_prices, open_prices, valid_tickers,
        initial_capital=INITIAL_CAPITAL,
        params=PARAMS,
        show_progress=True,
    )

    if result.equity.empty:
        print("ERROR: Empty equity curve.")
        return

    # 3. Report
    strat_metrics = compute_metrics(result.equity)
    spy_metrics = compute_metrics(result.spy_equity)

    print("\n" + format_comparison_table(strat_metrics, spy_metrics))
    print(f"\n  Trades: {len(result.trade_log)}")

    # Count trade types
    buy_count = sum(1 for t in result.trade_log if t["action"] == "buy")
    sell_count = sum(1 for t in result.trade_log if t["action"] == "sell")
    trim_count = sum(1 for t in result.trade_log if t["action"] == "trim")
    print(f"  Buys: {buy_count}  Sells: {sell_count}  Trims: {trim_count}")

    # 4. Save artifacts
    output_dir = create_output_dir("backtest")
    print(f"\nSaving to {output_dir}/")

    (output_dir / "equity.csv").write_text(result.equity.to_csv(), encoding="utf-8")
    save_equity_png(
        result.equity, result.spy_equity,
        output_dir,
        title="Hybrid Backtest (R\u00b2 Momentum + Vol-Targeting)",
        filename="equity_curve.png",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
