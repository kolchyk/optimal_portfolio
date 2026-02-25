"""Data models for simulation results."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class SimulationResult:
    """Complete output from a portfolio simulation run."""

    equity: pd.Series
    spy_equity: pd.Series
    holdings_history: pd.DataFrame  # DatetimeIndex x tickers, values = share counts
    cash_history: pd.Series  # daily cash balance
    regime_history: pd.Series | None  # daily bool: True = bull, False = bear (None when regime filter disabled)
    trade_log: list[dict] = field(default_factory=list)
    # Each entry: {"date": date, "ticker": str, "action": "buy"|"sell"|"liquidate",
    #              "shares": float, "price": float}
