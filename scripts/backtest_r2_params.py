"""R² Momentum backtest с фиксированными параметрами WFO.

Запуск бэктеста с рекомендованными параметрами из WFO (walk-forward optimization).

Параметры:
  r2_lookback=80, kama_asset=20, kama_spy=30, kama_buffer=0.03,
  gap=0.175, atr=30, top_n=5, rebal=4w

Usage:
    uv run python scripts/backtest_r2_params.py
    uv run python scripts/backtest_r2_params.py --period 5y
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.compare_methods import run_backtest
from src.portfolio_sim.cli_utils import create_output_dir, filter_valid_tickers, setup_logging
from src.portfolio_sim.config import INITIAL_CAPITAL, SPY_TICKER
from src.portfolio_sim.data import fetch_etf_tickers, fetch_price_data
from src.portfolio_sim.reporting import compute_metrics, format_comparison_table, save_equity_png


# Рекомендованные параметры из WFO (output от 2026-02-28)
R2_LOOKBACK = 80
KAMA_ASSET_PERIOD = 20
KAMA_SPY_PERIOD = 30
KAMA_BUFFER = 0.03
GAP_THRESHOLD = 0.175
ATR_PERIOD = 30
TOP_N = 5
REBAL_PERIOD_WEEKS = 4
RISK_FACTOR = 0.001


def main() -> None:
    parser = argparse.ArgumentParser(
        description="R² Momentum backtest с фиксированными WFO-параметрами",
    )
    parser.add_argument("--period", default="3y", help="yfinance period (default: 3y)")
    parser.add_argument("--refresh", action="store_true", help="Force refresh data cache")
    args = parser.parse_args()

    setup_logging()

    print("=" * 60)
    print("R² Momentum Backtest — WFO Recommended Params")
    print("=" * 60)
    print(f"\nParams: r2_lb={R2_LOOKBACK} kama_asset={KAMA_ASSET_PERIOD} "
          f"kama_spy={KAMA_SPY_PERIOD} kama_buf={KAMA_BUFFER} "
          f"gap={GAP_THRESHOLD} atr={ATR_PERIOD} top_n={TOP_N} rebal={REBAL_PERIOD_WEEKS}w")

    # 1. Load data
    print("\nLoading data...")
    tickers = fetch_etf_tickers()
    close_prices, open_prices = fetch_price_data(
        tickers, period=args.period, refresh=args.refresh, cache_suffix="_etf",
    )

    min_days = max(R2_LOOKBACK, KAMA_SPY_PERIOD, KAMA_ASSET_PERIOD) + 10
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
    equity, selection_history, trade_log = run_backtest(
        close_prices, open_prices, valid_tickers,
        initial_capital=INITIAL_CAPITAL,
        top_n=TOP_N,
        rebal_period_weeks=REBAL_PERIOD_WEEKS,
        kama_asset_period=KAMA_ASSET_PERIOD,
        kama_spy_period=KAMA_SPY_PERIOD,
        kama_buffer=KAMA_BUFFER,
        r2_lookback=R2_LOOKBACK,
        gap_threshold=GAP_THRESHOLD,
        atr_period=ATR_PERIOD,
        risk_factor=RISK_FACTOR,
        kama_cache_ext=None,
        spy_kama_ext=None,
    )

    if equity.empty:
        print("ERROR: Empty equity curve.")
        return

    # 3. SPY benchmark
    spy_close = close_prices[SPY_TICKER].reindex(equity.index).ffill()
    if spy_close.iloc[0] > 0:
        spy_equity = INITIAL_CAPITAL * (spy_close / spy_close.iloc[0])
    else:
        spy_equity = pd.Series(INITIAL_CAPITAL, index=equity.index)

    # 4. Report
    strat_metrics = compute_metrics(equity)
    spy_metrics = compute_metrics(spy_equity)

    print("\n" + format_comparison_table(strat_metrics, spy_metrics))
    print(f"\n  Trades: {len(trade_log)}")

    # 5. Save artifacts
    output_dir = create_output_dir("backtest_r2")
    print(f"\nSaving to {output_dir}/")

    (output_dir / "equity.csv").write_text(equity.to_csv(), encoding="utf-8")
    save_equity_png(
        equity, spy_equity,
        output_dir,
        title="R² Momentum Backtest (WFO params)",
        filename="equity_curve.png",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
