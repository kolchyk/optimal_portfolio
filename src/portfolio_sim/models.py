"""Data models for simulation results."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.portfolio_sim.params import StrategyParams


@dataclass
class SimulationResult:
    """Complete output from a portfolio simulation run."""

    equity: pd.Series
    spy_equity: pd.Series
    holdings_history: pd.DataFrame  # DatetimeIndex x tickers, values = share counts
    cash_history: pd.Series  # daily cash balance
    trade_log: list[dict] = field(default_factory=list)
    # Each entry: {"date": date, "ticker": str, "action": "buy"|"sell"|"liquidate",
    #              "shares": float, "price": float}


@dataclass
class WFOStep:
    """One step of walk-forward optimization."""

    step_index: int
    is_start: pd.Timestamp
    is_end: pd.Timestamp
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp
    optimized_params: StrategyParams
    is_metrics: dict
    oos_metrics: dict
    oos_equity: pd.Series  # raw OOS equity curve (starts at initial_capital)
    oos_spy_equity: pd.Series  # SPY equity over same OOS period


@dataclass
class WFOResult:
    """Complete walk-forward optimization result."""

    steps: list[WFOStep]
    stitched_equity: pd.Series  # concatenated OOS equity curves
    stitched_spy_equity: pd.Series  # concatenated SPY equity for same OOS periods
    oos_metrics: dict  # metrics computed on stitched OOS equity
    final_params: StrategyParams  # params from the last IS window (for live use)


@dataclass
class WFOGridEntry:
    """One (lookback_period, oos_days) combination in schedule grid search."""

    lookback_period: int
    oos_days: int
    wfo_result: WFOResult
    oos_calmar: float


@dataclass
class WFOGridResult:
    """Grid search over WFO schedule parameters (unified search space)."""

    entries: list[WFOGridEntry]
    best_entry: WFOGridEntry
    summary: pd.DataFrame  # rows = combos, cols = lookback_period, oos_days, calmar, cagr, maxdd
