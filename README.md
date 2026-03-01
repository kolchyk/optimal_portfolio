# Optimal Portfolio — Smart Momentum Strategy

## What Is This?

This is an **automated investment strategy** that picks the best-performing assets
(stocks, bonds, gold, real estate, crypto) and holds them while their trend is strong.
When the trend weakens, it sells and moves to cash. Think of it as a **smart autopilot
for your portfolio**.

**In simple terms:** instead of buying everything and hoping for the best,
this strategy only buys assets that are going UP in a smooth, reliable way,
and sells them when they start going DOWN. It's like surfing — you only ride
waves that are clean and strong.

## How It Works

1. **Look at 37 assets** across stocks, bonds, gold, real estate, and crypto
2. **Score each asset** by how smooth and strong its upward trend is (R² Momentum)
3. **Pick the top 5–10** with the best scores
4. **Size positions** so each asset contributes similar risk (ATR Risk Parity)
5. **Hold until the trend breaks** — then sell and look for replacements
6. **Scale the portfolio** to maintain steady risk (Vol-Targeting overlay)

The strategy checks the portfolio every few weeks, not every day.
This keeps trading costs low and avoids overreacting to noise.

---

## Asset Universe

37 tickers across 7 asset classes (tactical all-weather allocation):

| Asset Class | Count | Examples |
|---|---|---|
| US Equity | 11 | AAPL, MSFT, NVDA, GOOGL, QQQ |
| US Sector ETFs | 6 | XLK, XLF, XLE, SMH |
| International | 5 | VEA, VWO, ASML, TSM, MELI |
| Bonds | 6 | TLT, IEF, SHY, LQD, HYG, EMB |
| Metals & Commodities | 4 | GLD, SLV, CPER, LIT |
| Real Estate | 3 | VNQ, VNQI, REM |
| Crypto ETFs | 2 | IBIT, ETHA |

**What does this mean?** By investing across all these classes, the strategy
can profit in different market conditions — stocks may fall while bonds rise,
or gold may surge during uncertainty. The strategy dynamically picks the best
performers, wherever they are.

Benchmark: **S&P 500 (SPY)** — buy-and-hold.

---

## How the Strategy Picks Assets

### R² Momentum Scoring (Clenow method)

For each asset, we fit a regression line to its log-prices over the last N days:

```
score = annualized_return × R²
```

- **annualized_return** — how fast the asset is trending up (annualized)
- **R²** — how smooth the trend is (0 to 1). R² = 0.9 means very smooth, R² = 0.3 means choppy

**In plain English:** two assets may both be up 20%, but the one with a smooth, steady
climb gets a higher score than the one that zigzags wildly.

### Filter Pipeline

Before an asset enters the portfolio, it must pass these checks:

1. **SPY regime filter** — is the overall market in bull mode? (SPY > its KAMA trend)
2. **Asset trend filter** — is this specific asset trending up? (price > its KAMA)
3. **Gap filter** — no single-day jumps larger than the threshold (avoids distorted data)
4. **Correlation filter** — not too similar to assets already in the portfolio
5. **R² Momentum Score > 0** — positive and stable trend

### Vol-Targeting Overlay

The portfolio is scaled daily: `scale = target_vol / realised_vol`,
capped at `max_leverage`. This automatically reduces exposure in volatile
periods and increases it in calm ones.

**In plain English:** when markets get scary, the strategy automatically
reduces your exposure. When things are calm, it takes more risk.

### Position Sizing — ATR Risk Parity

```
weight[i] = (risk_factor / (ATR[i] / price[i])) / Σ(...)
```

Assets with low volatility (like bonds) get more capital.
Assets with high volatility (like crypto) get less.

**In plain English:** every position "risks" roughly the same amount of money per day,
regardless of whether it's a calm bond or a volatile tech stock.

---

## Strategy Parameters

### Core Parameters

| Parameter | Default | WFO Range | What It Does |
|---|---|---|---|
| `r2_lookback` | 90 | 60–120 (step 20) | How many days to look back for trend scoring |
| `kama_asset_period` | 10 | 10, 20, 30, 40, 50 | KAMA window for individual asset trend filter |
| `kama_spy_period` | 40 | 20, 30, 40, 50 | KAMA window for SPY regime filter |
| `kama_buffer` | 0.005 | 0.005–0.03 (step 0.005) | Buffer to prevent false bull/bear switches |
| `top_n` | 5 | 5–15 (step 5) | Max positions in portfolio |
| `rebal_period_weeks` | 3 | 2–4 (step 1) | How often we review the portfolio (weeks) |

### Vol-Targeting Parameters

| Parameter | Default | WFO Range | What It Does |
|---|---|---|---|
| `target_vol` | 0.10 (10%) | 0.05–0.20 | Target annual portfolio volatility |
| `max_leverage` | 1.5 | 1.0, 1.25, 1.5, 2.0 | Maximum scaling factor |
| `portfolio_vol_lookback` | 21 | 15–35 (step 10) | Window for estimating realized vol |
| `corr_threshold` | 0.7 | 0.5–1.0 (step 0.1) | Correlation filter for new entries |

### Fixed Parameters

| Parameter | Value | What It Does |
|---|---|---|
| `INITIAL_CAPITAL` | $10,000 | Starting capital |
| `COMMISSION_RATE` | 0.02% (2 bps) | Broker commission per trade |
| `SLIPPAGE_RATE` | 0.05% (5 bps) | Expected price slippage |
| `RISK_FREE_RATE` | 4% annual | For Sharpe Ratio calculation |
| `risk_factor` | 0.001 (10 bps) | Daily risk per position |

---

## Performance Metrics Explained

| Metric | Formula | What It Means |
|---|---|---|
| Total Return | equity_end / equity_start − 1 | Your total profit or loss as a percentage |
| CAGR | (equity_end / equity_start)^(252/days) − 1 | Average annual growth rate |
| Max Drawdown | max(peak − trough) / peak | Worst peak-to-bottom decline — the most you could have lost |
| Sharpe Ratio | (CAGR − risk_free) / ann_vol | Return per unit of risk. Above 1.0 is good, above 2.0 is excellent |
| Calmar Ratio | CAGR / max_drawdown | Annual return divided by worst drawdown |
| Annualized Vol | std(daily_returns) × √252 | How much the portfolio bounces around per year |
| Win Rate | days with return > 0 / total days | Percentage of profitable days |

---

## Usage

### CLI: Walk-Forward Optimization

```bash
# Install dependencies
uv sync

# Run WFO (default parameters)
uv run python -m src.portfolio_sim walk-forward

# With custom parameters
uv run python -m src.portfolio_sim walk-forward \
    --period 3y \
    --n-trials 100 \
    --n-workers 8 \
    --refresh
```

| Flag | Default | What It Does |
|---|---|---|
| `--period` | 3y | Data period (yfinance format) |
| `--n-trials` | 100 | Optuna trials per IS step |
| `--n-workers` | cpu_count − 1 | Parallel workers |
| `--refresh` | — | Force data re-download |

### Streamlit Dashboard

```bash
uv run streamlit run app.py
```

Two pages:

- **Backtest** — interactive backtesting with parameter controls via sidebar sliders
- **How It Works** — beginner-friendly strategy documentation

---

## Tech Stack

| Technology | Purpose |
|---|---|
| **Python 3.12+** | Language |
| **Numba** | JIT compilation for KAMA/ER (100x speedup) |
| **Optuna** | Bayesian parameter optimization (TPE sampler) |
| **Pandas / NumPy** | Time series data processing |
| **yfinance** | Market data download |
| **Streamlit** | Interactive dashboard |
| **Parquet** | Data cache (fast, compact) |
| **ProcessPoolExecutor** | Parallelism (bypasses GIL) |

---

## Project Structure

```
src/portfolio_sim/
├── cli.py              # CLI dispatcher
├── command.py          # CLI handler for walk-forward command
├── config.py           # Parameters, asset universe, search space
├── engine.py           # Hybrid bar-by-bar simulation engine
├── params.py           # StrategyParams (frozen dataclass)
├── parallel.py         # Parallel execution utilities (ProcessPoolExecutor)
├── optimizer.py        # KAMA precomputation, objective functions
├── walk_forward.py     # WFO schedule generation, walk-forward runner
├── models.py           # SimulationResult, WFOStep, WFOResult
├── data.py             # Data loading (yfinance + Parquet cache)
├── indicators.py       # KAMA and ER (Numba JIT)
├── reporting.py        # Metrics, charts, reports
└── cli_utils.py        # CLI utilities

scripts/
└── backtest_s2.py      # Backtest with fixed parameters
```
