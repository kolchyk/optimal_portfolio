"""Walk-Forward Optimization for R² Momentum — thin wrapper.

Implementation moved to src/portfolio_sim/walk_forward.py.
Run: uv run python scripts/wfo_r2_momentum.py
"""

# Re-export for backward compatibility
from src.portfolio_sim.engine import run_backtest  # noqa: F401
from src.portfolio_sim.models import R2WFOResult, R2WFOStep  # noqa: F401
from src.portfolio_sim.params import R2StrategyParams  # noqa: F401
from src.portfolio_sim.walk_forward import (  # noqa: F401
    format_r2_wfo_report,
    main,
    run_r2_walk_forward,
)

if __name__ == "__main__":
    main()
