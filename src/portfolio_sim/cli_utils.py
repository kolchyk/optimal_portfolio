"""Shared CLI utilities for entry-point scripts.

Consolidates logging setup, output directory creation, and ticker
filtering that was previously duplicated across run_*.py files.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import structlog


def setup_logging(level: int = logging.INFO) -> None:
    """Configure structlog with console rendering."""
    logging.basicConfig(level=level, format="%(message)s")
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.stdlib.LoggerFactory(),
    )


def create_output_dir(prefix: str) -> Path:
    """Create a timestamped output directory under ``output/``."""
    dt = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output") / f"{prefix}_{dt}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def filter_valid_tickers(
    close_prices: pd.DataFrame,
    min_days: int,
) -> list[str]:
    """Return tickers that have at least *min_days* non-NaN price rows."""
    return [
        t for t in close_prices.columns
        if len(close_prices[t].dropna()) >= min_days
    ]
