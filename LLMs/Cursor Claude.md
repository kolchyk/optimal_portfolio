# Improvement Plan: R2 Momentum Strategy

## Context

The R2 Momentum strategy (Clenow-style) underperforms SPY Buy-and-Hold for approximately half the backtest period and sits below the efficient frontier on the risk-return chart. Root cause analysis identifies three structural problems:

1. **Binary regime filter** — the SPY KAMA filter switches between 100% invested and 100% cash, causing the strategy to miss rebounds and incur whipsaw costs
2. **WFO overfitting** — the objective maximizes raw CAGR with a wide parameter search space, producing unstable parameters that perform well in-sample but degrade out-of-sample
3. **Momentum clustering** — the R2 engine lacks a correlation filter, so top-N positions cluster in the same sector and draw down together

## Improvements (ordered by impact/complexity ratio)

### 1. Switch WFO Objective from CAGR to Calmar Ratio
**Files:** [wfo_r2_momentum.py:85-99](scripts/wfo_r2_momentum.py#L85-L99)

Replace raw CAGR maximization with Calmar ratio (CAGR / MaxDD). Raw CAGR in a 126-day IS window rewards concentrated bets in whatever worked recently. Calmar penalizes drawdown, producing parameters more robust OOS.

```python
def r2_objective(equity, max_dd_limit=0.30, min_n_days=20):
    metrics = compute_metrics(equity)
    if metrics["n_days"] < min_n_days: return -999.0
    if metrics["max_drawdown"] > max_dd_limit: return -999.0
    cagr = metrics["cagr"]
    if cagr <= 0: return -999.0
    calmar = cagr / max(metrics["max_drawdown"], 0.01)
    return calmar
```

### 2. Constrain WFO Search Space
**Files:** [config.py:154-163](src/portfolio_sim/config.py#L154-L163)

Narrow ranges to prevent the optimizer from picking extreme values that overfit:
- `r2_lookback`: `[60, 80, 100]` (drop 20, 40 — too short for stable OLS regression)
- `top_n`: `[5, 10, 15]` (drop 20, 25 — too diluted)
- `rebal_period_weeks`: `[2, 3, 4]` (drop 1 and 5-6 — whipsaw or too infrequent)

### 3. Graduated Regime Filter (replace binary SPY KAMA)
**Files:** [compare_methods.py:88-101](scripts/compare_methods.py#L88-L101), [compare_methods.py:193-271](scripts/compare_methods.py#L193-L271), [compare_methods.py:428-492](scripts/compare_methods.py#L428-L492)

This is the **highest-impact structural change**. Replace the all-in/all-cash binary regime filter with graduated exposure:

- SPY >> KAMA: 100% invested (full risk-on)
- SPY near KAMA (buffer zone): 25%-100% linearly interpolated
- SPY << KAMA: 25% floor (keep top positions, rest in cash)

The 25% floor prevents missing the first 5-10 days of a rebound — this is the primary source of "underperformance for half the period".

Implementation:
- `is_risk_on() -> bool` becomes `compute_regime_exposure() -> float` (0.25 to 1.0)
- `select_r2_assets` scales target weights by exposure factor
- `run_backtest` handles partial liquidation instead of full liquidation
- Add `regime_floor` parameter (default 0.25) to `R2StrategyParams`

### 4. Add Correlation Filter to R2 Engine
**Files:** [compare_methods.py:193-271](scripts/compare_methods.py#L193-L271), [wfo_r2_momentum.py:57-68](scripts/wfo_r2_momentum.py#L57-L68)

V1/V2 engines already have correlation filtering, but the R2 engine has none. Add a greedy correlation filter after ranking:

```python
# In select_r2_assets, after ranking:
filtered = []
for t in ranked:
    if not filtered:
        filtered.append(t)
        continue
    recent_rets = returns[filtered + [t]].iloc[-lookback:]
    max_corr = recent_rets.corr()[t].drop(t).max()
    if max_corr < corr_threshold:
        filtered.append(t)
    if len(filtered) >= top_n:
        break
```

Add `corr_threshold` (default 0.7) to `R2StrategyParams` and `R2_SEARCH_SPACE`.

### 5. Daily KAMA Stop-Loss Monitoring Between Rebalances
**Files:** [compare_methods.py:347-492](scripts/compare_methods.py#L347-L492)

Currently the R2 engine only checks signals on rebalance dates (every 2-4 weeks). Between rebalances, a KAMA breach goes undetected. V1 engine checks daily. Port this to R2:

- Check KAMA stops for all held positions daily (not just on rebalance dates)
- Check SPY regime daily
- Only *buy* decisions remain on rebalance schedule (low turnover for entries)
- *Sell* decisions happen immediately when KAMA triggers (capital protection)

### 6. Integrate Vol-Targeting Overlay from V2
**Files:** [compare_methods.py:277-516](scripts/compare_methods.py#L277-L516), [wfo_r2_momentum.py:57-68](scripts/wfo_r2_momentum.py#L57-L68)

Port the V2 vol-targeting overlay (`target_vol / realized_vol` scaling) into the R2 engine while keeping the graduated regime filter. This adds a second layer of defense:
- Regime filter: controls base exposure level (macro)
- Vol-targeting: scales position sizes within that exposure (micro)

Add parameters: `target_vol` (default 0.12), `max_leverage` (1.0, no leverage), `portfolio_vol_lookback` (21 days).

### 7. Momentum Decay Timer for Lazy-Hold Positions
**Files:** [compare_methods.py:440-492](scripts/compare_methods.py#L440-L492)

Track position entry cycle. After `MAX_LAZY_CYCLES` (e.g. 6 rebalance cycles), require the position to still be in the top-N ranking to be kept. Stale winners that rode momentum 6 months ago but now rank #30 get exited even if their KAMA holds, making room for fresh momentum leaders.

## Verification Plan

After each improvement:
1. Run `uv run python scripts/compare_methods.py` — compare CAGR, Sharpe, MaxDD vs SPY
2. Run `streamlit run app.py` — visually verify equity curve, risk-return scatter (should move toward/above frontier)
3. Run `uv run python scripts/wfo_r2_momentum.py --n-trials 30 --period 3y` — verify OOS degradation improves

Key metrics to track:
- Sharpe ratio (target: > SPY Sharpe)
- Position on risk-return scatter (target: on or above efficient frontier)
- % of period outperforming SPY (target: > 60%)
- IS/OOS CAGR degradation (target: < 50%)
