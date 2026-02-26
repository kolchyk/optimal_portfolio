#!/usr/bin/env python3
"""
Custom simulation runner for specified parameters:
kama=30, lbk=20, top_n=5, buf=0.005
"""

import pandas as pd
from datetime import datetime
from pathlib import Path

from src.portfolio_sim.config import INITIAL_CAPITAL, DEFAULT_OUTPUT_DIR
from src.portfolio_sim.data import fetch_price_data, fetch_etf_tickers
from src.portfolio_sim.engine import run_simulation
from src.portfolio_sim.params import StrategyParams
from src.portfolio_sim.reporting import compute_metrics, save_equity_png, format_comparison_table

def main():
    # 1. Setup parameters
    params = StrategyParams(
        kama_period=30,
        lookback_period=20,
        top_n=5,
        kama_buffer=0.005,
    )
    
    # 2. Prepare output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = DEFAULT_OUTPUT_DIR / f"custom_sim_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"--- Starting Custom Simulation ---")
    print(f"Params: KAMA={params.kama_period}, Lookback={params.lookback_period}, Top-N={params.top_n}, Buffer={params.kama_buffer}")
    # 3. Load data
    tickers_requested = fetch_etf_tickers()
    close_prices, open_prices = fetch_price_data(tickers_requested, period="5y", cache_suffix="_etf")
    
    # Filter tickers to only those available in close_prices
    tickers = [t for t in tickers_requested if t in close_prices.columns]
    print(f"Tickers: {len(tickers_requested)} requested, {len(tickers)} available")
    
    # 4. Run simulation
    print("Running simulation...")
    result = run_simulation(
        close_prices=close_prices,
        open_prices=open_prices,
        tickers=tickers,
        initial_capital=INITIAL_CAPITAL,
        params=params,
        show_progress=True
    )
    
    # 5. Save Equity PNG
    print("Saving equity curve...")
    save_equity_png(
        equity=result.equity,
        spy_equity=result.spy_equity,
        output_dir=output_dir,
        title=f"Custom Sim: KAMA 30/20/5/0.005"
    )
    
    # 6. Save Transactions CSV
    print("Saving transactions...")
    if result.trade_log:
        trades_df = pd.DataFrame(result.trade_log)
        # Add trade value for convenience
        trades_df["value"] = trades_df["shares"] * trades_df["price"]
        trades_df.to_csv(output_dir / "transactions.csv", index=False)
    else:
        print("No trades executed.")
        (output_dir / "transactions.csv").write_text("No trades executed.")
    
    # 7. Compute metrics and save summary
    print("Computing metrics and saving summary...")
    strat_metrics = compute_metrics(result.equity)
    spy_metrics = compute_metrics(result.spy_equity)
    
    comparison_table = format_comparison_table(strat_metrics, spy_metrics)
    
    final_value = result.equity.iloc[-1]
    total_return_pct = (final_value / INITIAL_CAPITAL - 1) * 100
    
    summary_text = [
        "CUSTOM SIMULATION SUMMARY",
        "=" * 40,
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Parameters: KAMA={params.kama_period}, Lookback={params.lookback_period}, Top-N={params.top_n}, Buffer={params.kama_buffer}",
        f"Initial Capital: ${INITIAL_CAPITAL:,.2f}",
        f"Final Portfolio Value: ${final_value:,.2f} ({total_return_pct:+.2f}%)",
        "",
        comparison_table
    ]
    
    summary_path = output_dir / "summary.txt"
    summary_path.write_text("\n".join(summary_text))
    
    print(f"\nSimulation complete!")
    print(f"Final Value: ${final_value:,.2f}")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
