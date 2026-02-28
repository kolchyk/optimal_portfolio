# AGENTS.md

## Learned User Preferences

- Respond to the user in Russian; the user writes exclusively in Russian (occasionally Ukrainian). Do not switch to English unless the user does.
- Always use `uv` for Python execution: `uv run python script.py` and `uv run streamlit run dashboard.py`. Never use bare `python` or `streamlit` commands.
- "Запусти проект" means run the Streamlit dashboard: `uv run streamlit run dashboard.py`.
- All tickers come from `src/portfolio_sim/config.py` (`ETF_UNIVERSE`). Never fetch tickers from Wikipedia, CSV files, or any external source.
- SPY is benchmark only — it is not part of the tradable universe. The S&P 500 universe was entirely removed from the codebase.
- Never change `COMMISSION_RATE = 0.0002` (2 bps) or `SLIPPAGE_RATE = 0.0005` (5 bps) in `config.py` without explicit user instruction.
- Limit `yfinance` data fetches to 5 years of history. Do not download longer periods.
- Cache price data in `output/cache/`; download tickers in batches of 100.
- Do not modify Optuna parameter search ranges without explicit user direction. Current confirmed ranges in `R2_SEARCH_SPACE`: `r2_lookback` 20–120 step 20; `kama_asset_period` [10,20,30,40,50]; `kama_spy_period` [20,30,40,50]; `kama_buffer` 0.005–0.03 step 0.005; `gap_threshold` 0.10–0.20 step 0.025; `atr_period` 10–30 step 5; `top_n` 5–25 step 5; `rebal_period_weeks` 1–6 step 1.
- Three strategy bugs that must never be reintroduced: (1) Rank-21 Trap — sell only when KAMA stop-loss triggers or score < 0, not when a stock drops out of top-N (Lazy Hold); (2) SPY Dribble — use directional thresholds with `KAMA_BUFFER` for bull/bear hysteresis; (3) All-in Sizing — use ATR risk parity sizing, not total-cash divided by number of buys.
- Only one optimization mode exists: Walk-Forward Optimization (WFO). Sensitivity Analysis (standalone) and Max-Profit Search have been removed.
- When shown a plan, trim unnecessary scope before implementing rather than building everything proposed.
- Prefer concise plain language; avoid verbose documentation and duplicated sections.
- Streamlit width: use `width="stretch"` instead of `use_container_width=True`; use `width="content"` instead of `use_container_width=False`.

## Learned Workspace Facts

- Project: `optimal_portfolio` — R² Momentum Strategy backtester and optimizer (Clenow-style: regression scoring, KAMA trend filters, ATR risk parity). GitHub: `https://github.com/kolchyk/optimal_portfolio`.
- Tech stack: Python 3.12+, `uv` package manager (`pyproject.toml` + `uv.lock`), `yfinance`, `optuna`, `streamlit`, `pandas`, `numpy`, `numba`. Deployed on Heroku via `Procfile`.
- `src/portfolio_sim/config.py` is the single source of truth for tickers (`ETF_UNIVERSE`, `ASSET_CLASS_MAP`), fixed strategy parameters (`KAMA_PERIOD`, `KAMA_SPY_PERIOD`, `TOP_N`, `KAMA_BUFFER`, `R2_LOOKBACK`, `GAP_THRESHOLD`, `ATR_PERIOD`, `RISK_FACTOR`, `REBAL_PERIOD_WEEKS`, `COMMISSION_RATE`, `SLIPPAGE_RATE`, `INITIAL_CAPITAL`, `RISK_FREE_RATE`), and optimization search space (`R2_SEARCH_SPACE`).
- Entry point for the dashboard: `uv run streamlit run app.py` (multi-page Streamlit app with `pages/` subdirectory).
- Key CLI: `uv run python -m src.portfolio_sim walk-forward` (Walk-Forward Optimization, the only optimization mode).
- R² engine: `scripts/compare_methods.py` (`run_backtest`). WFO: `scripts/wfo_r2_momentum.py` (`run_r2_walk_forward`).
- Output directories (all gitignored): `output/cache/` (Parquet price cache), `output/sim_*/`, `output/wfo_*/`, `output/backtest_r2_*/`.
- Test runner: `uv run pytest`; coverage: `uv run pytest --cov=src/portfolio_sim`. Tests live in `tests/`.
- Optimization objective: `cagr` (CAGR maximization) with MaxDD ≤ 30% guard. WFO uses Optuna TPE sampler per IS step.
- `src/portfolio_sim/parallel.py` provides parallel execution utilities; `src/portfolio_sim/cli_utils.py` provides shared CLI argument helpers.
