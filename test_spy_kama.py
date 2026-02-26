import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.portfolio_sim.indicators import compute_kama_series
from src.portfolio_sim.reporting import compute_metrics
from src.portfolio_sim.config import KAMA_BUFFER

def compute_strat_returns(close, returns, period, buffer):
    """Compute strategy returns with hysteresis signal logic."""
    kama = compute_kama_series(close, period=period)
    
    # Hysteresis signal generation (stateful)
    signals = np.zeros(len(close))
    is_bull = False
    
    first_valid_idx = kama.first_valid_index()
    if first_valid_idx is None:
        return pd.Series(0.0, index=close.index)
        
    start_pos = close.index.get_loc(first_valid_idx)
    if close.iloc[start_pos] > kama.iloc[start_pos]:
        is_bull = True
    
    for i in range(start_pos, len(close)):
        price = close.iloc[i]
        k_val = kama.iloc[i]
        
        if pd.isna(k_val):
            signals[i] = 0
            continue
            
        if is_bull:
            if price < k_val * (1 - buffer):
                is_bull = False
        else:
            if price > k_val * (1 + buffer):
                is_bull = True
        
        signals[i] = 1 if is_bull else 0
        
    signals_shifted = pd.Series(signals, index=close.index).shift(1).fillna(0)
    return returns * signals_shifted

def compute_strat_equity(close, returns, period, buffer, initial_capital):
    strat_returns = compute_strat_returns(close, returns, period, buffer)
    return initial_capital * (1 + strat_returns).cumprod()

def run_test():
    print("--- SPY KAMA Trend Filter Test (3 Years) ---")
    
    # 1. Download data
    ticker = "SPY"
    print(f"Downloading {ticker} data...")
    data = yf.download(ticker, period="5y", interval="1d")
    if data.empty:
        print("Error: No data downloaded.")
        return

    # Use 'Close' prices
    # yfinance returns a MultiIndex if one ticker is passed sometimes, or just a Series.
    # To be safe, let's ensure we have a Series.
    if isinstance(data.columns, pd.MultiIndex):
        close = data['Close'][ticker]
    else:
        close = data['Close']
    
    # Calculate daily returns
    returns = close.pct_change().fillna(0)
    
    # Initial capital
    initial_capital = 10000.0
    
    # Results dictionary
    results = {}
    
    # --- Option 1: Buy & Hold SPY ---
    spy_equity = initial_capital * (1 + returns).cumprod()
    results["Buy & Hold SPY"] = spy_equity
    
    # --- Option 2: 2D Grid Search (Period & Buffer) ---
    periods = [10, 20, 30, 40, 50, 60, 80, 100]
    buffer_values = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]
    
    print(f"\nPerforming 2D Grid Search (Periods: {len(periods)}, Buffers: {len(buffer_values)})...")
    
    grid_results = []
    for p in periods:
        for b in buffer_values:
            equity = compute_strat_equity(
                close, returns, period=p, buffer=b, initial_capital=initial_capital
            )
            m = compute_metrics(equity)
            grid_results.append({
                "period": p,
                "buffer": b,
                "equity": equity,
                "metrics": m
            })
            
    best_by_sharpe = max(grid_results, key=lambda x: x["metrics"]["sharpe"])
    print(f"Optimal Strategy: Period {best_by_sharpe['period']}, Buffer {best_by_sharpe['buffer']:.1%} (Sharpe: {best_by_sharpe['metrics']['sharpe']:.2f})")
    
    results[f"KAMA {best_by_sharpe['period']} (Buf {best_by_sharpe['buffer']:.1%}) [Optimum]"] = best_by_sharpe["equity"]
    
    # --- Option 3: Walk-Forward Optimization (WFO) ---
    # 3 months lookback (approx 63 trading days)
    # 1 month test (approx 21 trading days)
    lookback_days = 63
    test_days = 21
    
    print(f"\nPerforming Walk-Forward Optimization (3m lookback, 1m test)...")
    
    wfo_returns = pd.Series(0.0, index=close.index)
    selected_params = []
    
    # Start WFO after enough data for lookback + some warmup for KAMA
    # Let's say we start after 126 days (~6 months)
    start_idx = 126
    
    for i in range(start_idx, len(close) - test_days, test_days):
        train_start = i - lookback_days
        train_end = i
        test_start = i
        test_end = i + test_days
        
        # Optimize on train slice
        train_close = close.iloc[train_start:train_end]
        train_returns = returns.iloc[train_start:train_end]
        
        best_p, best_b, best_s = 10, 0.0, -np.inf
        for p in periods:
            for b in buffer_values:
                # Need some extra context before train_start to compute KAMA correctly
                # compute_kama_series handles this by using the passed slice
                s_returns = compute_strat_returns(train_close, train_returns, p, b)
                # Compute Sharpe for this slice
                # Simple Sharpe: mean / std (annualized later)
                if s_returns.std() > 0:
                    sr = s_returns.mean() / s_returns.std()
                else:
                    sr = -1.0
                
                if sr > best_s:
                    best_s, best_p, best_b = sr, p, b
        
        selected_params.append({"period": best_p, "buffer": best_b})
        
        # Apply to test slice
        # To avoid look-ahead bias and handle KAMA state correctly:
        # We need the full history UP TO test_end to compute the signal, but only keep returns for the test window.
        full_up_to_test = compute_strat_returns(close.iloc[:test_end], returns.iloc[:test_end], best_p, best_b)
        wfo_returns.iloc[test_start:test_end] = full_up_to_test.iloc[test_start:test_end]

    wfo_equity = initial_capital * (1 + wfo_returns).cumprod()
    results["KAMA WFO (3m/1m)"] = wfo_equity
    
    # Analyze Parameter Stability
    param_df = pd.DataFrame(selected_params)
    most_stable = param_df.groupby(['period', 'buffer']).size().reset_index(name='count').sort_values('count', ascending=False)
    print("\nMost Stable Parameters (WFO chosen frequency):")
    print(most_stable.head(3).to_string(index=False))

    # --- Metrics ---
    metrics_list = []
    for name, equity in results.items():
        m = compute_metrics(equity)
        metrics_list.append({
            "Strategy": name,
            "Total Return": f"{m['total_return']:.1%}",
            "CAGR": f"{m['cagr']:.1%}",
            "Max DD": f"{m['max_drawdown']:.1%}",
            "Sharpe": f"{m['sharpe']:.2f}"
        })
    
    # Print metrics table
    df_metrics = pd.DataFrame(metrics_list)
    # Sort by Total Return descending
    df_metrics['tr_val'] = df_metrics['Total Return'].str.rstrip('%').astype(float)
    df_metrics = df_metrics.sort_values(by='tr_val', ascending=False).drop(columns=['tr_val'])
    
    print("\nPerformance Comparison:")
    print(df_metrics.to_string(index=False))
    
    # --- Plotting 1: Optimized Result vs B&H ---
    plt.figure(figsize=(12, 7))
    for name, equity in results.items():
        linewidth = 2.5 if "Buy & Hold" in name or "[Optimum]" in name or "WFO" in name else 1.2
        alpha = 1.0 if "Buy & Hold" in name or "[Optimum]" in name or "WFO" in name else 0.6
        plt.plot(equity, label=name, linewidth=linewidth, alpha=alpha)
        
    plt.title(f"SPY vs KAMA Optimized (WFO & Fixed)", fontsize=14, fontweight="bold")
    plt.ylabel("Equity ($)")
    plt.xlabel("Date")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    plot_path1 = output_dir / "spy_kama_test.png"
    plt.savefig(plot_path1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nEquity chart saved to: {plot_path1}")

    # --- Plotting 2: Buffer Sensitivity for Optimal Period ---
    opt_p = best_by_sharpe['period']
    plt.figure(figsize=(12, 7))
    plt.plot(spy_equity, label="Buy & Hold SPY", linewidth=2.5, color="black")
    
    # Filter grid results for the optimal period to show buffer sensitivity
    for res in grid_results:
        if res['period'] == opt_p:
            b = res['buffer']
            label = f"Buf {b:.1%}"
            linewidth = 2.0 if b == best_by_sharpe['buffer'] else 1.0
            alpha = 1.0 if b == best_by_sharpe['buffer'] else 0.4
            plt.plot(res['equity'], label=label, linewidth=linewidth, alpha=alpha)
            
    plt.title(f"Buffer Sensitivity for Optimal KAMA Period {opt_p}", fontsize=14, fontweight="bold")
    plt.ylabel("Equity ($)")
    plt.xlabel("Date")
    plt.legend(ncol=2)
    plt.grid(True, alpha=0.3)
    
    plot_path2 = output_dir / "spy_kama_optimized.png"
    plt.savefig(plot_path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Buffer sensitivity chart saved to: {plot_path2}")

if __name__ == "__main__":
    run_test()
