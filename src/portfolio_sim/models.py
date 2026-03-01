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
    oos_equity: pd.Series
    oos_spy_equity: pd.Series


@dataclass
class WFOResult:
    """Complete walk-forward optimization result."""

    steps: list[WFOStep]
    stitched_equity: pd.Series
    stitched_spy_equity: pd.Series
    oos_metrics: dict
    final_params: StrategyParams
