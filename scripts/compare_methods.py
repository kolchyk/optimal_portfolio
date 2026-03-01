"""Clenow R² Momentum Strategy — thin wrapper.

Implementation moved to src/portfolio_sim/engine.py.
Run: uv run python scripts/compare_methods.py
"""

# Re-export for backward compatibility
from src.portfolio_sim.engine import run_backtest  # noqa: F401
from src.portfolio_sim.visualization import main

if __name__ == "__main__":
    main()
