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
