## Tree for src
```
├── __init__.py
└── portfolio_sim/
    ├── config.py
    ├── optimization.py
    ├── __init__.py
    ├── indicators.py
    ├── engine.py
    ├── alpha.py
    ├── walk_forward.py
    ├── data.py
    └── reporting.py
```

## File: portfolio_sim/config.py
```python
"""Typed configuration for portfolio simulation.

All constants are non-tunable. Only StrategyParams fields are optimized by Optuna.
"""

from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Non-tunable constants
# ---------------------------------------------------------------------------
INITIAL_CAPITAL: float = 10_000
COMMISSION_RATE: float = 0.001  # 0.1% per trade
SLIPPAGE_RATE: float = 0.0015  # 15 bps market impact on Open auctions
MAX_WEIGHT: float = 0.15  # Maximum single-position weight
BREADTH_THRESHOLD: float = 0.30  # Market breadth "kill switch"
VOL_FLOOR: float = 0.05  # Minimum annualized volatility for weighting
REBALANCE_INTERVAL: int = 21  # Trading days between rebalances
RISK_FREE_RATE: float = 0.04
SAFE_HAVEN_TICKER: str = "SHV"

# Walk-Forward Validation
WFV_TRAIN_DAYS: int = 252  # 1 year
WFV_TEST_DAYS: int = 63  # 1 quarter

# Paths
TICKERS_JSON_PATH: Path = Path("portfolio_tickers.json")
DEFAULT_OUTPUT_DIR: Path = Path("output")
CACHE_DIR: Path = DEFAULT_OUTPUT_DIR / "cache"


# ---------------------------------------------------------------------------
# Tunable parameters (4 total — Occam's Razor)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StrategyParams:
    """The 4 tunable parameters for the Simple Baseline strategy."""

    kama_period: int = 20  # KAMA indicator period
    lookback_period: int = 126  # Momentum lookback (trading days)
    max_correlation: float = 0.70  # Correlation constraint for diversification
    top_n_selection: int = 15  # Max number of positions


# Optuna search space (mirrors StrategyParams fields)
PARAM_SPACE: dict = {
    "kama_period": {"type": "int", "low": 10, "high": 40, "step": 10},
    "lookback_period": {"type": "int", "low": 63, "high": 126, "step": 21},
    "max_correlation": {"type": "float", "low": 0.5, "high": 0.9, "step": 0.1},
    "top_n_selection": {"type": "int", "low": 10, "high": 25, "step": 5},
}
```
## File: portfolio_sim/optimization.py
```python
"""Optuna-based hyperparameter optimization with multiprocessing."""

import multiprocessing
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from src.portfolio_sim.config import (
    DEFAULT_OUTPUT_DIR,
    INITIAL_CAPITAL,
    RISK_FREE_RATE,
    StrategyParams,
)
from src.portfolio_sim.engine import run_simulation


def objective(
    trial: optuna.Trial,
    sim_prices: pd.DataFrame,
    sim_open: pd.DataFrame,
    full_prices: pd.DataFrame,
    tickers: list[str],
    metric: str = "calmar",
) -> float:
    """Optuna objective: maximize the chosen metric."""
    params = StrategyParams(
        kama_period=trial.suggest_int("kama_period", 10, 40, step=10),
        lookback_period=trial.suggest_int("lookback_period", 63, 126, step=21),
        max_correlation=trial.suggest_float("max_correlation", 0.5, 0.9, step=0.1),
        top_n_selection=trial.suggest_int("top_n_selection", 10, 25, step=5),
    )

    try:
        equity, _, _ = run_simulation(
            sim_prices, sim_open, full_prices, tickers, params, INITIAL_CAPITAL
        )

        if not equity or np.isnan(equity[-1]):
            return -1.0

        eq = pd.Series(equity)
        days = len(eq)

        cagr = (eq.iloc[-1] / eq.iloc[0]) ** (252 / max(1, days)) - 1
        rolling_max = eq.cummax()
        drawdown = (eq - rolling_max) / rolling_max
        max_dd = abs(drawdown.min())

        if metric == "sharpe":
            returns = eq.pct_change().dropna()
            ann_vol = returns.std() * np.sqrt(252)
            if ann_vol == 0:
                return -1.0
            return float((cagr - RISK_FREE_RATE) / ann_vol)
        elif metric == "return":
            return float(cagr)
        else:  # calmar (default)
            if max_dd == 0:
                return -1.0
            calmar = cagr / max_dd
            if max_dd > 0.25:
                calmar *= 0.1  # Penalty for drawdown > 25%
            return float(calmar)
    except Exception:
        return -1.0


def _worker(
    storage_url: str,
    study_name: str,
    sim_prices_path: str,
    sim_open_path: str,
    full_prices_path: str,
    tickers: list[str],
    n_trials: int,
    metric: str,
) -> None:
    """Worker process for parallel Optuna optimization."""
    sim_prices = pd.read_parquet(sim_prices_path)
    sim_open = pd.read_parquet(sim_open_path)
    full_prices = pd.read_parquet(full_prices_path)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.load_study(study_name=study_name, storage=storage_url)
    study.optimize(
        lambda trial: objective(
            trial, sim_prices, sim_open, full_prices, tickers, metric
        ),
        n_trials=n_trials,
        n_jobs=1,
    )


def run_optimization(
    sim_prices: pd.DataFrame,
    sim_open: pd.DataFrame,
    full_prices: pd.DataFrame,
    tickers: list[str],
    n_trials: int = 100,
    metric: str = "calmar",
    output_dir: Path | None = None,
) -> optuna.Study:
    """Run Optuna optimization using all CPU cores via multiprocessing.

    Returns the completed Study object.
    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Optuna Optimization ({n_trials} trials, metric: {metric}) ---")
    n_cpu = multiprocessing.cpu_count()
    n_workers = min(n_cpu, n_trials)
    print(f"Using {n_workers} worker processes...")

    db_path = output_dir / "optuna_study.db"
    storage_url = f"sqlite:///{db_path}"
    study_name = f"portfolio_opt_{metric}"

    if db_path.exists():
        db_path.unlink()

    optuna.create_study(
        direction="maximize",
        storage=storage_url,
        study_name=study_name,
        load_if_exists=False,
    )

    # Serialize DataFrames for spawn-safe transfer
    tmp_dir = output_dir / "tmp_opt"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    sim_path = tmp_dir / "sim_prices.parquet"
    sim_open_path = tmp_dir / "sim_open.parquet"
    full_path = tmp_dir / "full_prices.parquet"
    sim_prices.to_parquet(sim_path)
    sim_open.to_parquet(sim_open_path)
    full_prices.to_parquet(full_path)

    # Distribute trials across workers
    trials_per_worker = [n_trials // n_workers] * n_workers
    for idx in range(n_trials % n_workers):
        trials_per_worker[idx] += 1

    processes = []
    for idx in range(n_workers):
        p = multiprocessing.Process(
            target=_worker,
            args=(
                storage_url,
                study_name,
                str(sim_path),
                str(sim_open_path),
                str(full_path),
                tickers,
                trials_per_worker[idx],
                metric,
            ),
        )
        p.start()
        processes.append(p)

    # Monitor progress
    start_time = time.time()
    while any(p.is_alive() for p in processes):
        try:
            study = optuna.load_study(study_name=study_name, storage=storage_url)
            completed = len(study.trials)
            elapsed = time.time() - start_time
            if completed > 0:
                speed = completed / elapsed
                remaining = (n_trials - completed) / speed if speed > 0 else 0
                print(
                    f"  Progress: {completed}/{n_trials} "
                    f"({completed / n_trials:.0%}) | "
                    f"{speed:.1f} trials/sec | "
                    f"ETA: {remaining:.0f}s",
                    end="\r",
                )
        except Exception:
            pass
        time.sleep(2)

    for p in processes:
        p.join()

    # Cleanup temp files
    sim_path.unlink(missing_ok=True)
    sim_open_path.unlink(missing_ok=True)
    full_path.unlink(missing_ok=True)
    if tmp_dir.exists() and not any(tmp_dir.iterdir()):
        tmp_dir.rmdir()

    study = optuna.load_study(study_name=study_name, storage=storage_url)

    print(f"\n--- Optimization complete ---")
    if study.best_trials:
        print(f"Best {metric}: {study.best_value:.4f}")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")

    return study
```
## File: portfolio_sim/indicators.py
```python
"""Kaufman's Adaptive Moving Average (KAMA) — Numba JIT accelerated.

Standalone implementation (no external config dependencies).
"""

import numpy as np
import pandas as pd
from numba import njit


@njit(cache=True)
def _kama_recurrent_loop(
    data_array: np.ndarray,
    smoothing_array: np.ndarray,
    kama: np.ndarray,
    period: int,
) -> np.ndarray:
    """Numba JIT recurrent KAMA update. cache=True for zero warm-up on restart."""
    n = len(data_array)
    for i in range(period + 1, n):
        smoothing = smoothing_array[i - period]
        kama[i] = kama[i - 1] + smoothing * (data_array[i] - kama[i - 1])
    return kama


def compute_kama(
    data: np.ndarray | pd.Series,
    period: int = 20,
    fast_constant: int = 2,
    slow_constant: int = 30,
) -> np.ndarray:
    """Compute Kaufman Adaptive Moving Average.

    Returns np.ndarray of KAMA values (NaN-padded at the start).
    """
    period = max(1, int(period))
    fast_constant = max(1, int(fast_constant))
    slow_constant = max(1, int(slow_constant))

    data_array = np.asarray(data, dtype=float)
    n = len(data_array)

    if n <= period:
        return np.full(n, np.nan, dtype=float)

    kama = np.full(n, np.nan, dtype=float)

    fast_const = 2.0 / (fast_constant + 1)
    slow_const = 2.0 / (slow_constant + 1)

    # Vectorized price change: abs(data[i] - data[i - period])
    price_change = np.abs(data_array[period:] - data_array[:-period])

    # Vectorized volatility: rolling sum of abs(diff(data)) over period
    abs_diff = np.abs(np.diff(data_array))
    abs_diff_cumsum = np.concatenate(([0], np.cumsum(abs_diff)))
    volatility = abs_diff_cumsum[period:] - abs_diff_cumsum[:-period]

    # Efficiency Ratio
    with np.errstate(divide="ignore", invalid="ignore"):
        er = np.where(volatility > 0, price_change / volatility, 0.0)
    er = np.clip(er, 0, 1)

    # Smoothing constant
    sc = (er * (fast_const - slow_const) + slow_const) ** 2
    smoothing_array = np.clip(sc, 0, 1)

    # Initial KAMA value
    kama[period] = data_array[period]

    # Recurrent loop via Numba JIT
    kama = _kama_recurrent_loop(data_array, smoothing_array, kama, period)

    return kama


def compute_kama_series(prices: pd.Series, period: int = 20) -> pd.Series:
    """Convenience wrapper returning pd.Series with the same index as input."""
    kama_arr = compute_kama(prices.values, period=period)
    return pd.Series(kama_arr, index=prices.index)
```
## File: portfolio_sim/engine.py
```python
"""Bar-by-bar simulation engine.

Signals computed on Close(T), execution on Open(T+1).
Includes Market Breadth filter (Step 1), KAMA trailing stops, and SHV parking.
"""

import numpy as np
import pandas as pd

from src.portfolio_sim.alpha import compute_target_weights
from src.portfolio_sim.config import (
    BREADTH_THRESHOLD,
    COMMISSION_RATE,
    REBALANCE_INTERVAL,
    SAFE_HAVEN_TICKER,
    SLIPPAGE_RATE,
    StrategyParams,
)
from src.portfolio_sim.indicators import compute_kama_series


def run_simulation(
    sim_prices: pd.DataFrame,
    sim_open: pd.DataFrame,
    full_prices: pd.DataFrame,
    tickers: list[str],
    params: StrategyParams,
    initial_capital: float,
) -> tuple[list[float], list[float], np.ndarray]:
    """Run a full bar-by-bar portfolio simulation.

    Args:
        sim_prices: Close prices for the simulation period.
        sim_open: Open prices for the simulation period (same index).
        full_prices: Full history of Close prices (for lookback/indicators).
        tickers: master list of tradable tickers (including SHV).
        params: the 4 tunable strategy parameters.
        initial_capital: starting cash.

    Returns:
        equity: daily portfolio values (Close mark-to-market).
        long_exposures: daily long exposure ratio (excl SHV).
        final_weights: last-applied target weights array.
    """
    lookback = params.lookback_period

    equity: list[float] = []
    long_exposures: list[float] = []
    val_alpha = initial_capital

    # ------------------------------------------------------------------
    # 0. Pre-compute KAMA for all tickers on full history
    # ------------------------------------------------------------------
    kama_cache: dict[str, pd.Series] = {}
    for t in tickers + ["SPY"]:
        if t in full_prices.columns:
            kama_cache[t] = compute_kama_series(
                full_prices[t].dropna(), period=params.kama_period
            )

    # ------------------------------------------------------------------
    # 1. Initial signal (on Close of the day before sim start)
    # ------------------------------------------------------------------
    idx_start = full_prices.index.get_loc(sim_prices.index[0])
    past_prices_init = full_prices.iloc[idx_start - lookback : idx_start][tickers]

    prev_date = full_prices.index[idx_start - 1]
    kama_init = {
        t: kama_cache[t].get(prev_date, 0)
        for t in tickers
        if t in kama_cache and not np.isnan(kama_cache[t].get(prev_date, np.nan))
    }

    pending_weights = compute_target_weights(
        past_prices_init, tickers, params, kama_init
    )

    # Park remainder in SHV
    total_w = pending_weights.sum()
    if total_w < 1.0 and SAFE_HAVEN_TICKER in tickers:
        safe_idx = tickers.index(SAFE_HAVEN_TICKER)
        pending_weights[safe_idx] += 1.0 - total_w

    shares: dict[str, float] = {t: 0.0 for t in tickers}
    current_weights = np.zeros(len(tickers))

    # ------------------------------------------------------------------
    # 2. Day-by-day loop
    # ------------------------------------------------------------------
    for i, (date, daily_prices) in enumerate(sim_prices.iterrows()):
        open_prices = sim_open.loc[date].fillna(daily_prices)

        # --- Execute pending orders on Open ---
        if pending_weights is not None:
            val_at_open = sum(shares[t] * open_prices[t] for t in tickers)
            if val_at_open <= 0:
                val_at_open = val_alpha

            turnover = 0.0
            new_shares: dict[str, float] = {}
            tolerance = 0.02

            for j, t in enumerate(tickers):
                target_w = pending_weights[j]
                current_w = (
                    (shares[t] * open_prices[t]) / val_at_open
                    if val_at_open > 0
                    else 0.0
                )

                if abs(target_w - current_w) > tolerance or target_w == 0:
                    target_val = val_at_open * target_w
                    current_val = shares[t] * open_prices[t]
                    turnover += abs(target_val - current_val)
                    new_shares[t] = target_val / open_prices[t] if open_prices[t] > 0 else 0.0
                else:
                    new_shares[t] = shares[t]

            val_alpha = val_at_open - turnover * (COMMISSION_RATE + SLIPPAGE_RATE)
            val_alpha = max(0.0, val_alpha)

            multiplier = (val_alpha / val_at_open) if val_at_open > 1e-6 else 0.0
            shares = {t: new_shares[t] * multiplier for t in tickers}
            current_weights = pending_weights
            pending_weights = None

        # --- Mark-to-market on Close ---
        val_alpha = sum(shares[t] * daily_prices[t] for t in tickers)

        if np.isnan(val_alpha) or val_alpha <= 0.001:
            val_alpha = 0.0
            remaining = len(sim_prices) - len(equity)
            equity.extend([0.0] * remaining)
            long_exposures.extend([0.0] * remaining)
            break

        equity.append(val_alpha)

        # Long exposure (excl SHV)
        denom = val_alpha if abs(val_alpha) > 1e-6 else 1.0
        gross_long = (
            sum(
                shares[t] * daily_prices[t]
                for t in tickers
                if t != SAFE_HAVEN_TICKER
            )
            / denom
        )
        long_exposures.append(gross_long)

        # ------------------------------------------------------------------
        # 3. Compute signals on Close(T) for execution on Open(T+1)
        # ------------------------------------------------------------------

        # Step 1: Market Breadth filter
        active_uptrends = 0
        total_valid = 0
        for t in tickers:
            if t == SAFE_HAVEN_TICKER:
                continue
            t_kama = kama_cache.get(t, pd.Series(dtype=float)).get(date, np.nan)
            if not np.isnan(t_kama):
                total_valid += 1
                if daily_prices[t] > t_kama:
                    active_uptrends += 1

        breadth = active_uptrends / max(1, total_valid)
        is_bull = breadth >= BREADTH_THRESHOLD

        if not is_bull:
            # All to SHV
            new_weights = np.zeros(len(tickers))
            if SAFE_HAVEN_TICKER in tickers:
                safe_idx = tickers.index(SAFE_HAVEN_TICKER)
                new_weights[safe_idx] = 1.0
            pending_weights = new_weights
            continue

        # Individual KAMA trailing stops
        should_rebalance = i > 0 and i % REBALANCE_INTERVAL == 0

        stop_tickers: list[str] = []
        for t in tickers:
            if t == SAFE_HAVEN_TICKER or shares[t] <= 0:
                continue
            t_kama = kama_cache.get(t, pd.Series(dtype=float)).get(date, np.nan)
            if not np.isnan(t_kama) and daily_prices[t] < t_kama:
                stop_tickers.append(t)

        if should_rebalance:
            # Full rebalance
            idx_in_full = full_prices.index.get_loc(date)
            past_prices = full_prices.iloc[
                idx_in_full - lookback + 1 : idx_in_full + 1
            ][tickers]

            kama_current = {
                t: kama_cache[t].get(date, 0)
                for t in tickers
                if t in kama_cache
                and not np.isnan(kama_cache[t].get(date, np.nan))
            }

            new_weights = compute_target_weights(
                past_prices, tickers, params, kama_current
            )

            total_invested = new_weights.sum()
            if total_invested < 1.0 and SAFE_HAVEN_TICKER in tickers:
                safe_idx = tickers.index(SAFE_HAVEN_TICKER)
                new_weights[safe_idx] += 1.0 - total_invested

            pending_weights = new_weights

        elif stop_tickers:
            # Partial rebalance: exit stopped positions to SHV
            new_weights = np.zeros(len(tickers))
            freed_weight = 0.0

            for j, t in enumerate(tickers):
                w = (
                    (shares[t] * daily_prices[t]) / val_alpha
                    if val_alpha > 0
                    else 0.0
                )
                if t in stop_tickers:
                    freed_weight += w
                else:
                    new_weights[j] = w

            if SAFE_HAVEN_TICKER in tickers and freed_weight > 0:
                safe_idx = tickers.index(SAFE_HAVEN_TICKER)
                new_weights[safe_idx] += freed_weight

            total = new_weights.sum()
            if total > 0:
                new_weights /= total

            pending_weights = new_weights

    return equity, long_exposures, current_weights
```
## File: portfolio_sim/alpha.py
```python
"""Simple Baseline strategy — the production alpha.

5-step algorithm (Steps 2-5 here; Step 1 Market Breadth is in engine.py):
  2. KAMA trend filter   — keep only tickers with Close > KAMA
  3. Raw momentum ranking — score = Close[-1] / Close[-lookback] - 1
  4. Correlation walk-down — greedy selection, skip if corr > max_correlation
  5. Inverse volatility weighting — w ~ 1 / annualized_vol, cap at MAX_WEIGHT
"""

import numpy as np
import pandas as pd

from src.portfolio_sim.config import MAX_WEIGHT, SAFE_HAVEN_TICKER, VOL_FLOOR, StrategyParams


def compute_target_weights(
    prices_window: pd.DataFrame,
    tickers: list[str],
    params: StrategyParams,
    kama_values: dict[str, float],
) -> np.ndarray:
    """Compute target portfolio weights for the given price window.

    Args:
        prices_window: Close prices, rows = lookback_period trading days, cols = tickers.
        tickers: ordered list of ticker symbols matching engine's master list.
        params: the 4 tunable strategy parameters.
        kama_values: {ticker: current_kama_value} for KAMA entry filter.

    Returns:
        np.ndarray of weights aligned with *tickers*. Sums to <= 1.0.
        The caller (engine) allocates the remainder (1 - sum) to SHV.
    """
    # Step 2: KAMA trend filter
    candidates = [
        t
        for t in tickers
        if t != SAFE_HAVEN_TICKER
        and t in prices_window.columns
        and not np.isnan(prices_window[t].iloc[-1])
        and prices_window[t].iloc[-1] > kama_values.get(t, 0)
    ]

    if not candidates:
        return np.zeros(len(tickers))

    # Step 3: Raw 6-month momentum ranking
    momentum: dict[str, float] = {}
    for t in candidates:
        close_now = prices_window[t].iloc[-1]
        close_past = prices_window[t].iloc[0]
        if close_past > 1e-8:
            momentum[t] = close_now / close_past - 1
        else:
            momentum[t] = 0.0

    # Sort descending, keep only positive momentum
    ranked = sorted(
        [(t, s) for t, s in momentum.items() if s > 0],
        key=lambda x: x[1],
        reverse=True,
    )

    if not ranked:
        return np.zeros(len(tickers))

    # Step 4: Correlation walk-down
    candidate_tickers = [t for t, _ in ranked]
    returns_df = prices_window[candidate_tickers].pct_change().dropna()
    corr_matrix = returns_df.corr().fillna(0)

    selected: list[str] = []
    for t, _ in ranked:
        if len(selected) >= params.top_n_selection:
            break
        if not any(
            abs(corr_matrix.loc[t, s]) > params.max_correlation for s in selected
        ):
            selected.append(t)

    if not selected:
        return np.zeros(len(tickers))

    # Step 5: Inverse volatility weighting
    recent_returns = prices_window[selected].pct_change().iloc[-20:]
    vol = (recent_returns.std() * np.sqrt(252)).clip(lower=VOL_FLOOR)

    raw_weights = 1.0 / vol
    raw_weights = raw_weights / raw_weights.sum()

    # Cap at MAX_WEIGHT
    if len(selected) * MAX_WEIGHT < 1.0:
        # Not enough positions to fill 100% at MAX_WEIGHT — just clip.
        # Remainder will be allocated to SHV by the engine.
        raw_weights = raw_weights.clip(upper=MAX_WEIGHT)
    else:
        # Redistribute excess iteratively
        for _ in range(10):
            capped = raw_weights > MAX_WEIGHT
            if not capped.any():
                break
            excess = raw_weights[capped].sum() - capped.sum() * MAX_WEIGHT
            raw_weights[capped] = MAX_WEIGHT
            not_capped = ~capped
            if not_capped.any() and raw_weights[not_capped].sum() > 1e-8:
                raw_weights[not_capped] += (
                    raw_weights[not_capped] / raw_weights[not_capped].sum()
                ) * excess
            else:
                break

    # Map back to full ticker array
    result = np.zeros(len(tickers))
    for t in selected:
        idx = tickers.index(t)
        result[idx] = raw_weights[t]

    return result
```
## File: portfolio_sim/walk_forward.py
```python
"""Walk-Forward Validation orchestrator.

WFV is the only production mode for evaluating strategy performance.
Train on 1 year, test (blind) on 1 quarter, slide forward by 1 quarter.
"""

import pandas as pd
import structlog

from src.portfolio_sim.config import (
    INITIAL_CAPITAL,
    WFV_TEST_DAYS,
    WFV_TRAIN_DAYS,
    StrategyParams,
)
from src.portfolio_sim.engine import run_simulation
from src.portfolio_sim.optimization import run_optimization

log = structlog.get_logger(__name__)


def generate_wfv_windows(
    total_days: int,
    train_days: int = WFV_TRAIN_DAYS,
    test_days: int = WFV_TEST_DAYS,
    buffer_days: int = 0,
) -> list[tuple[int, int, int, int]]:
    """Generate sliding window indices.

    Returns list of (train_start, train_end, test_start, test_end).
    """
    windows = []
    cursor = buffer_days
    while cursor + train_days + test_days <= total_days:
        train_start = cursor
        train_end = cursor + train_days
        test_start = train_end
        test_end = train_end + test_days
        windows.append((train_start, train_end, test_start, test_end))
        cursor += test_days
    return windows


def run_walk_forward(
    prices: pd.DataFrame,
    open_prices: pd.DataFrame,
    tickers: list[str],
    n_trials: int = 100,
    metric: str = "calmar",
    train_days: int = WFV_TRAIN_DAYS,
    test_days: int = WFV_TEST_DAYS,
) -> dict:
    """Walk-Forward Validation: optimize on train, blind test on test.

    Returns dict with:
        oos_equity: pd.Series — chained OOS equity curve
        windows: list[dict] — per-window results (params, dates, metrics)
        oos_segments: list[pd.Series] — individual OOS equity segments
        is_segments: list[pd.Series] — individual IS equity segments
    """
    default_params = StrategyParams()
    buffer_days = default_params.lookback_period + 10
    total_days = len(prices)

    windows = generate_wfv_windows(total_days, train_days, test_days, buffer_days)
    if not windows:
        raise ValueError(
            f"Not enough data for WFV: {total_days} days, "
            f"need at least {buffer_days + train_days + test_days}"
        )

    print(f"\n{'=' * 60}")
    print(f"Walk-Forward Validation: {len(windows)} windows")
    print(f"Train: {train_days}d | Test: {test_days}d | Buffer: {buffer_days}d")
    print(f"{'=' * 60}")

    oos_equities: list[pd.Series] = []
    is_equities: list[pd.Series] = []
    window_results: list[dict] = []

    for w_idx, (train_start, train_end, test_start, test_end) in enumerate(windows):
        train_dates = prices.index[train_start:train_end]
        test_dates = prices.index[test_start:test_end]

        print(f"\n--- Window {w_idx + 1}/{len(windows)} ---")
        print(f"  Train: {train_dates[0].date()} -> {train_dates[-1].date()}")
        print(f"  Test:  {test_dates[0].date()} -> {test_dates[-1].date()}")

        sim_train = prices.iloc[train_start:train_end]
        sim_train_open = open_prices.iloc[train_start:train_end]
        full_for_train = prices.iloc[:train_end]

        # 1. Optimize on TRAIN
        print(f"  Optimizing ({n_trials} trials, metric={metric})...")
        study = run_optimization(
            sim_train,
            sim_train_open,
            full_for_train,
            tickers,
            n_trials=n_trials,
            metric=metric,
        )

        if study.best_trials:
            best_params = StrategyParams(**study.best_params)
            is_score = study.best_value
        else:
            print("  Optimization yielded no results, using defaults")
            best_params = StrategyParams()
            is_score = 0.0

        print(f"  IS best score: {is_score:.4f}")

        # IS equity (train data with best params)
        is_eq, _, _ = run_simulation(
            sim_train,
            sim_train_open,
            full_for_train,
            tickers,
            best_params,
            INITIAL_CAPITAL,
        )
        is_equities.append(pd.Series(is_eq, index=sim_train.index[: len(is_eq)]))

        # 2. Blind test on OOS
        sim_test = prices.iloc[test_start:test_end]
        sim_test_open = open_prices.iloc[test_start:test_end]
        full_for_test = prices.iloc[:test_end]

        oos_eq, _, _ = run_simulation(
            sim_test,
            sim_test_open,
            full_for_test,
            tickers,
            best_params,
            INITIAL_CAPITAL,
        )

        oos_series = pd.Series(oos_eq, index=sim_test.index[: len(oos_eq)])
        oos_equities.append(oos_series)

        # OOS window metrics
        if len(oos_eq) > 1 and oos_eq[0] > 0:
            oos_ret = (oos_eq[-1] / oos_eq[0] - 1) * 100
            rolling_max = pd.Series(oos_eq).cummax()
            oos_dd = ((pd.Series(oos_eq) - rolling_max) / rolling_max).min() * 100
        else:
            oos_ret = 0.0
            oos_dd = 0.0

        print(f"  OOS Return: {oos_ret:.1f}% | OOS MaxDD: {oos_dd:.1f}%")

        window_results.append(
            {
                "window": w_idx + 1,
                "train_start": str(train_dates[0].date()),
                "train_end": str(train_dates[-1].date()),
                "test_start": str(test_dates[0].date()),
                "test_end": str(test_dates[-1].date()),
                "params": {
                    "kama_period": best_params.kama_period,
                    "lookback_period": best_params.lookback_period,
                    "max_correlation": best_params.max_correlation,
                    "top_n_selection": best_params.top_n_selection,
                },
                "is_score": is_score,
                "oos_return_pct": oos_ret,
                "oos_max_dd_pct": oos_dd,
            }
        )

    # Chain OOS equity into continuous curve
    combined_oos = _chain_equity_segments(oos_equities, INITIAL_CAPITAL)

    print(f"\n{'=' * 60}")
    print("Walk-Forward Validation complete!")
    if len(combined_oos) > 1 and combined_oos.iloc[0] > 0:
        total_oos_ret = (combined_oos.iloc[-1] / combined_oos.iloc[0] - 1) * 100
        rolling_max = combined_oos.cummax()
        total_oos_dd = ((combined_oos - rolling_max) / rolling_max).min() * 100
        days = len(combined_oos)
        cagr = (combined_oos.iloc[-1] / combined_oos.iloc[0]) ** (
            252 / max(1, days)
        ) - 1
        print(f"OOS Total Return: {total_oos_ret:.1f}%")
        print(f"OOS CAGR: {cagr:.1%}")
        print(f"OOS Max Drawdown: {total_oos_dd:.1f}%")
    print(f"{'=' * 60}")

    return {
        "oos_equity": combined_oos,
        "windows": window_results,
        "oos_segments": oos_equities,
        "is_segments": is_equities,
    }


def _chain_equity_segments(
    segments: list[pd.Series], initial_capital: float
) -> pd.Series:
    """Chain OOS segments into a continuous equity curve.

    Each segment is scaled so its start equals the previous segment's end.
    """
    if not segments:
        return pd.Series(dtype=float)

    chained_values: list[float] = []
    chained_index: list = []
    current_capital = initial_capital

    for seg in segments:
        if seg.empty or seg.iloc[0] == 0:
            continue
        scale = current_capital / seg.iloc[0]
        scaled = seg * scale
        chained_values.extend(scaled.values.tolist())
        chained_index.extend(seg.index.tolist())
        current_capital = scaled.iloc[-1]

    return pd.Series(chained_values, index=chained_index)
```
## File: portfolio_sim/data.py
```python
"""Data loading: ticker universe from JSON, price fetching via yfinance, Parquet cache."""

import json

import pandas as pd
import structlog
import yfinance as yf

from src.portfolio_sim.config import CACHE_DIR, SAFE_HAVEN_TICKER, TICKERS_JSON_PATH

log = structlog.get_logger(__name__)

CLOSE_CACHE = CACHE_DIR / "close_prices.parquet"
OPEN_CACHE = CACHE_DIR / "open_prices.parquet"


def load_tickers(path=TICKERS_JSON_PATH) -> tuple[list[str], dict]:
    """Load ticker universe from JSON.

    Returns:
        (tickers_list, original_portfolio_dict)
    """
    with open(path) as f:
        data = json.load(f)
    tickers = list(set(data.get("tickers_600", [])))
    original_portfolio = data.get("original_portfolio", {})
    return tickers, original_portfolio


def fetch_price_data(
    tickers: list[str],
    period: str = "5y",
    refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch Close and Open prices for all tickers.

    Uses Parquet cache. Returns (close_prices, open_prices) DataFrames
    with DatetimeIndex rows and ticker columns.
    """
    if CLOSE_CACHE.exists() and OPEN_CACHE.exists() and not refresh:
        log.info("Loading prices from Parquet cache")
        close_df = pd.read_parquet(CLOSE_CACHE)
        open_df = pd.read_parquet(OPEN_CACHE)
        return close_df, open_df

    # Ensure SPY and SHV are included
    full_list = list(set(tickers + ["SPY", SAFE_HAVEN_TICKER]))
    log.info("Downloading prices via yfinance", n_tickers=len(full_list), period=period)

    close_df, open_df = _download_from_yfinance(full_list, period)

    # Save to cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    close_df.to_parquet(CLOSE_CACHE)
    open_df.to_parquet(OPEN_CACHE)
    log.info("Prices cached to Parquet", path=str(CACHE_DIR))

    return close_df, open_df


def _download_from_yfinance(
    tickers: list[str], period: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download OHLCV data via yfinance and extract Close/Open."""
    raw = yf.download(
        tickers,
        period=period,
        group_by="ticker",
        auto_adjust=True,
        threads=True,
        progress=True,
    )

    # yfinance returns MultiIndex columns (ticker, field) when multiple tickers
    if isinstance(raw.columns, pd.MultiIndex):
        close_df = raw.xs("Close", axis=1, level=1) if "Close" in raw.columns.get_level_values(1) else pd.DataFrame()
        open_df = raw.xs("Open", axis=1, level=1) if "Open" in raw.columns.get_level_values(1) else pd.DataFrame()
    else:
        # Single ticker fallback
        close_df = raw[["Close"]].rename(columns={"Close": tickers[0]})
        open_df = raw[["Open"]].rename(columns={"Open": tickers[0]})

    # Clean up
    close_df = close_df.ffill().dropna(axis=1, how="all")
    open_df = open_df[close_df.columns].ffill().bfill()

    # Ensure timezone-naive DatetimeIndex
    if close_df.index.tz is not None:
        close_df.index = close_df.index.tz_localize(None)
        open_df.index = open_df.index.tz_localize(None)

    log.info("Download complete", tickers_received=len(close_df.columns), rows=len(close_df))
    return close_df, open_df
```
## File: portfolio_sim/reporting.py
```python
"""Performance metrics, drawdown computation, and report generation."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.portfolio_sim.config import RISK_FREE_RATE


def compute_metrics(equity: pd.Series) -> dict:
    """Compute performance metrics for an equity curve.

    Returns dict with: total_return, cagr, max_drawdown, sharpe, calmar,
    annualized_vol, n_days.
    """
    if equity.empty or equity.iloc[0] <= 0:
        return {
            "total_return": 0.0,
            "cagr": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
            "calmar": 0.0,
            "annualized_vol": 0.0,
            "n_days": 0,
        }

    days = len(equity)
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (252 / max(1, days)) - 1

    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    max_dd = abs(drawdown.min())

    returns = equity.pct_change().dropna()
    ann_vol = returns.std() * np.sqrt(252)

    sharpe = (cagr - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else 0.0
    calmar = cagr / max_dd if max_dd > 0 else 0.0

    return {
        "total_return": float(total_return),
        "cagr": float(cagr),
        "max_drawdown": float(max_dd),
        "sharpe": float(sharpe),
        "calmar": float(calmar),
        "annualized_vol": float(ann_vol),
        "n_days": days,
    }


def compute_drawdown_series(equity: pd.Series) -> pd.Series:
    """Compute the underwater/drawdown series (values <= 0)."""
    if equity.empty:
        return pd.Series(dtype=float)
    rolling_max = equity.cummax()
    return (equity - rolling_max) / rolling_max


def format_metrics_table(metrics: dict) -> str:
    """Format metrics dict as a readable CLI table."""
    lines = [
        "Performance Metrics",
        "-" * 40,
        f"  Total Return:   {metrics['total_return']:>8.1%}",
        f"  CAGR:           {metrics['cagr']:>8.1%}",
        f"  Max Drawdown:   {metrics['max_drawdown']:>8.1%}",
        f"  Sharpe Ratio:   {metrics['sharpe']:>8.2f}",
        f"  Calmar Ratio:   {metrics['calmar']:>8.2f}",
        f"  Ann. Volatility:{metrics['annualized_vol']:>8.1%}",
        f"  Trading Days:   {metrics['n_days']:>8d}",
    ]
    return "\n".join(lines)


def save_wfv_report(wfv_result: dict, metric: str, output_dir: Path) -> Path:
    """Save Walk-Forward Validation markdown report. Returns path."""
    oos_equity = wfv_result["oos_equity"]
    windows = wfv_result["windows"]
    oos_metrics = compute_metrics(oos_equity)

    report = [
        "# Walk-Forward Validation Report",
        f"\n**OOS Period:** {oos_equity.index[0].strftime('%Y-%m-%d')} to "
        f"{oos_equity.index[-1].strftime('%Y-%m-%d')}",
        f"**Optimization Metric:** {metric.upper()}",
        f"**Windows:** {len(windows)}",
        "\n## OOS Performance (Blind Test)",
        f"\n- **Total Return:** {oos_metrics['total_return']:.1%}",
        f"- **CAGR:** {oos_metrics['cagr']:.1%}",
        f"- **Max Drawdown:** {oos_metrics['max_drawdown']:.1%}",
        f"- **Sharpe:** {oos_metrics['sharpe']:.2f}",
        f"- **Calmar:** {oos_metrics['calmar']:.2f}",
        "\n## Window Breakdown",
        "\n| Window | Train Period | Test Period | IS Score | OOS Return | OOS MaxDD |",
        "| :---: | :--- | :--- | :---: | :---: | :---: |",
    ]

    for w in windows:
        report.append(
            f"| {w['window']} | {w['train_start']} -> {w['train_end']} | "
            f"{w['test_start']} -> {w['test_end']} | "
            f"{w['is_score']:.4f} | {w['oos_return_pct']:.1f}% | "
            f"{w['oos_max_dd_pct']:.1f}% |"
        )

    report.append("\n## Parameters Per Window")
    for w in windows:
        report.append(f"\n### Window {w['window']}")
        report.append("\n| Parameter | Value |")
        report.append("| :--- | :--- |")
        for k, v in w["params"].items():
            report.append(f"| {k} | {v} |")

    report.append("\n---")
    report.append(
        f"\n*Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "wfv_report.md"
    path.write_text("\n".join(report))
    print(f"WFV report saved to {path}")
    return path


def save_wfv_json(wfv_result: dict, output_dir: Path) -> Path:
    """Save WFV window details as JSON. Returns path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "wfv_windows.json"
    with open(path, "w") as f:
        json.dump(wfv_result["windows"], f, indent=2, default=str)
    print(f"WFV JSON saved to {path}")
    return path
```
