"""Walk-Forward Optimization for V2 Vol-Targeted KAMA — thin wrapper.

Run: uv run python scripts/wfo_v2_momentum.py [options]
"""

import sys

from src.portfolio_sim.cli import main

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "walk-forward-v2"] + sys.argv[1:]
    main()
