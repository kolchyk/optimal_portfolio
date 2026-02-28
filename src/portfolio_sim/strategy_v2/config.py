"""V2 strategy configuration — search space and defaults for vol-targeted strategy."""

# ---------------------------------------------------------------------------
# V2 search space (extends base with vol-targeting parameters)
# ---------------------------------------------------------------------------
V2_SEARCH_SPACE = {
    # Base KAMA / momentum params
    "kama_period": {"type": "categorical", "choices": [10, 20, 30]},
    "lookback_period": {"type": "int", "low": 20, "high": 100, "step": 20},
    "kama_buffer": {"type": "float", "low": 0.005, "high": 0.03, "step": 0.005},
    "top_n": {"type": "int", "low": 3, "high": 15, "step": 3},
    "oos_days": {"type": "int", "low": 10, "high": 40, "step": 10},
    "corr_threshold": {"type": "float", "low": 0.5, "high": 0.95, "step": 0.05},
    # Force risk_parity — critical for vol targeting to work well
    "weighting_mode": {"type": "categorical", "choices": ["risk_parity"]},
    # Vol-targeting params (new)
    "target_vol": {"type": "float", "low": 0.06, "high": 0.15, "step": 0.01},
    "max_leverage": {"type": "categorical", "choices": [1.0, 1.25, 1.5]},
    "portfolio_vol_lookback": {"type": "int", "low": 15, "high": 36, "step": 7},
}

V2_PARAM_NAMES: list[str] = [
    "kama_period", "lookback_period", "kama_buffer", "top_n",
    "oos_days", "corr_threshold", "weighting_mode",
    "target_vol", "max_leverage", "portfolio_vol_lookback",
]

V2_DEFAULT_N_TRIALS: int = 200
V2_MAX_DD_LIMIT: float = 0.20  # tighter than v1's 0.30
