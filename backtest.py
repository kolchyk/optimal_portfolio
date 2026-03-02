"""Standalone backtester — run a single 3-year simulation with fixed WFO params.

Usage:
    uv run python backtest.py
"""

from pathlib import Path

import pandas as pd

from src.portfolio_sim.config import INITIAL_CAPITAL
from src.portfolio_sim.data import fetch_etf_tickers, fetch_price_data
from src.portfolio_sim.engine import run_simulation
from src.portfolio_sim.params import StrategyParams
from src.portfolio_sim.reporting import (
    compute_metrics,
    format_comparison_table,
    save_equity_png,
)

# ── Optimized stable parameters (Sharpe 2.29, stable across sub-periods) ──
PARAMS = StrategyParams(
    r2_window=30,
    kama_asset_period=13,
    kama_buffer=0.04,
    atr_period=20,
    risk_factor=0.001,
    top_n=10,
    rebal_days=18,
    max_per_class=5,
    target_vol=0.12,
    max_leverage=1.1,
    portfolio_vol_lookback=25,
    min_invested_pct=1.0,
)

PERIOD = "3y"
OUTPUT_DIR = Path("output/backtest")


def main() -> None:
    # 1. Load data
    tickers = fetch_etf_tickers()
    close, opn, high, low = fetch_price_data(
        tickers, period=PERIOD, cache_suffix="_etf",
    )
    print(f"Data: {close.index[0].date()} .. {close.index[-1].date()}  "
          f"({len(close)} bars, {len(close.columns)} tickers)")

    # 2. Run simulation
    result = run_simulation(
        close, opn, tickers, INITIAL_CAPITAL,
        params=PARAMS,
        show_progress=True,
        high_prices=high,
        low_prices=low,
    )

    # 3. Metrics
    strat_m = compute_metrics(result.equity)
    spy_m = compute_metrics(result.spy_equity)

    print()
    print(format_comparison_table(strat_m, spy_m))

    # 4. Save outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    result.equity.to_csv(OUTPUT_DIR / "equity.csv", header=True)
    result.spy_equity.to_csv(OUTPUT_DIR / "spy_equity.csv", header=True)

    save_equity_png(
        result.equity, result.spy_equity, OUTPUT_DIR,
        title="Backtest: WFO Recommended Params (3y)",
        filename="equity_curve.png",
    )

    trades_df = pd.DataFrame(result.trade_log)
    if not trades_df.empty:
        trades_df.to_csv(OUTPUT_DIR / "trades.csv", index=False)
        print(f"Trades: {len(trades_df)} total")

    print(f"\nOutputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
