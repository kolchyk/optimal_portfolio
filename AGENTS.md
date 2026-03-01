# AGENTS.md

## Learned User Preferences

- Respond to the user in Russian; the user writes exclusively in Russian (occasionally Ukrainian). Do not switch to English unless the user does.
- Always use `uv` for Python execution: `uv run python script.py` and `uv run streamlit run dashboard.py`. Never use bare `python` or `streamlit` commands.
- "Запусти проект" means run the Streamlit dashboard: `uv run streamlit run app.py`.
- All tickers come from `src/portfolio_sim/config.py` (`ETF_UNIVERSE`). Never fetch tickers from Wikipedia, CSV files, or any external source.
- SPY is benchmark only — it is not part of the tradable universe.
- Never change `COMMISSION_RATE = 0.0002` (2 bps) or `SLIPPAGE_RATE = 0.0005` (5 bps) in `config.py` without explicit user instruction.
- Limit `yfinance` data fetches to 5 years of history. Do not download longer periods.
- Cache price data in `output/cache/`; download tickers in batches of 100.
- Do not modify Optuna parameter search ranges without explicit user direction. Current confirmed ranges in `SEARCH_SPACE`: `r2_lookback` 60–120 step 20; `kama_asset_period` [10,20,30,40,50]; `kama_spy_period` [20,30,40,50]; `kama_buffer` 0.005–0.03 step 0.005; `gap_threshold` 0.10–0.20 step 0.025; `atr_period` 10–30 step 5; `top_n` 5–15 step 5; `rebal_period_weeks` 2–4 step 1; `corr_threshold` 0.5–1.0 step 0.1; `target_vol` 0.05–0.20 step 0.05; `max_leverage` [1.0, 1.25, 1.5, 2.0]; `portfolio_vol_lookback` 15–35 step 10.
- Three strategy bugs that must never be reintroduced: (1) Rank-21 Trap — sell only when KAMA stop-loss triggers or score < 0, not when a stock drops out of top-N (Lazy Hold); (2) SPY Dribble — use directional thresholds with `kama_buffer` for bull/bear hysteresis; (3) All-in Sizing — use ATR risk parity sizing, not total-cash divided by number of buys.
- Only one optimization mode exists: Walk-Forward Optimization (WFO).
- When shown a plan, trim unnecessary scope before implementing rather than building everything proposed.
- Prefer concise plain language; avoid verbose documentation and duplicated sections.
- Streamlit width: use `width="stretch"` instead of `use_container_width=True`; use `width="content"` instead of `use_container_width=False`.

## Learned Workspace Facts

- Project: `optimal_portfolio` — Hybrid R² Momentum + Vol-Targeting strategy backtester and optimizer (Clenow-style R² scoring, KAMA trend filters, ATR risk parity, vol-targeting overlay, correlation filter). GitHub: `https://github.com/kolchyk/optimal_portfolio`.
- Tech stack: Python 3.12+, `uv` package manager (`pyproject.toml` + `uv.lock`), `yfinance`, `optuna`, `streamlit`, `pandas`, `numpy`, `numba`. Deployed on Heroku via `Procfile`.
- `src/portfolio_sim/config.py` is the single source of truth for tickers (`ETF_UNIVERSE`, `ASSET_CLASS_MAP`), fixed constants (`COMMISSION_RATE`, `SLIPPAGE_RATE`, `INITIAL_CAPITAL`, `RISK_FREE_RATE`), and optimization search space (`SEARCH_SPACE`, `PARAM_NAMES`, `DEFAULT_N_TRIALS`, `MAX_DD_LIMIT`).
- `src/portfolio_sim/params.py` contains `StrategyParams` (frozen dataclass) with all strategy parameters (R² scoring + ATR risk parity + vol-targeting + correlation filter).
- Entry point for the dashboard: `uv run streamlit run app.py` (multi-page Streamlit app with `pages/` subdirectory).
- Key CLI: `uv run python -m src.portfolio_sim walk-forward` (Walk-Forward Optimization, the only optimization mode).
- Engine: `src/portfolio_sim/engine.py` (`run_simulation`) — hybrid bar-by-bar simulation.
- Output directories (all gitignored): `output/cache/` (Parquet price cache), `output/wfo_*/`.
- Test runner: `uv run pytest`; coverage: `uv run pytest --cov=src/portfolio_sim`. Tests live in `tests/`.
- Optimization objective: `total_return` (CAGR maximization) with MaxDD ≤ 25% guard. WFO uses Optuna TPE sampler per IS step.
- `src/portfolio_sim/parallel.py` provides parallel execution utilities; `src/portfolio_sim/cli_utils.py` provides shared CLI argument helpers.
