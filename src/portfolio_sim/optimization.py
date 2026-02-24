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
        kama_period=trial.suggest_int("kama_period", 5, 21, step=2),
        lookback_period=trial.suggest_int("lookback_period", 10, 40, step=5),
        top_n_selection=trial.suggest_int("top_n_selection", 5, 20, step=5),
    )

    try:
        equity, _, _, _ = run_simulation(
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
    storage_url = f"sqlite:///{db_path}?timeout=30"
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
