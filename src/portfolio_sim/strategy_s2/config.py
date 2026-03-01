"""S2 strategy configuration — search space and defaults for hybrid strategy."""

# ---------------------------------------------------------------------------
# S2 search space (R² Momentum + vol-targeting parameters)
# ---------------------------------------------------------------------------
S2_SEARCH_SPACE = {
    # R² Momentum params
    "r2_lookback": {"type": "int", "low": 60, "high": 120, "step": 20},
    "kama_asset_period": {"type": "categorical", "choices": [10, 20, 30, 40, 50]},
    "kama_spy_period": {"type": "categorical", "choices": [20, 30, 40, 50]},
    "kama_buffer": {"type": "float", "low": 0.005, "high": 0.03, "step": 0.005},
    "atr_period": {"type": "int", "low": 10, "high": 30, "step": 5},
    "top_n": {"type": "int", "low": 5, "high": 15, "step": 5},
    "rebal_period_weeks": {"type": "int", "low": 2, "high": 4, "step": 1},
    "gap_threshold": {"type": "float", "low": 0.10, "high": 0.20, "step": 0.025},
    "corr_threshold": {"type": "float", "low": 0.5, "high": 1.0, "step": 0.1},
    # Vol-targeting params
    "target_vol": {"type": "float", "low": 0.05, "high": 0.20, "step": 0.05},
    "max_leverage": {"type": "categorical", "choices": [1.0, 1.25, 1.5, 2.0]},
    "portfolio_vol_lookback": {"type": "int", "low": 15, "high": 35, "step": 10},
}

S2_PARAM_NAMES: list[str] = [
    "r2_lookback", "kama_asset_period", "kama_spy_period", "kama_buffer",
    "atr_period", "top_n", "rebal_period_weeks", "gap_threshold",
    "corr_threshold", "target_vol", "max_leverage", "portfolio_vol_lookback",
]

S2_DEFAULT_N_TRIALS: int = 100
S2_MAX_DD_LIMIT: float = 0.25


def get_s2_kama_periods(space: dict[str, dict] | None = None) -> list[int]:
    """Extract all possible KAMA period values from S2 search space."""
    space = space or S2_SEARCH_SPACE
    periods: list[int] = []
    for key in ("kama_asset_period", "kama_spy_period"):
        spec = space.get(key, {})
        if not spec:
            continue
        if spec.get("type") == "categorical":
            periods.extend(spec["choices"])
        else:
            periods.extend(range(
                spec.get("low", 10),
                spec.get("high", 40) + 1,
                spec.get("step", 5),
            ))
    return sorted(set(periods))
