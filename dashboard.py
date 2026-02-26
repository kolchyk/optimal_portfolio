"""Streamlit dashboard for KAMA momentum strategy.

Run with: streamlit run dashboard.py
"""

import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.optimize as sco
import streamlit as st
import structlog

from src.portfolio_sim.cli_utils import filter_valid_tickers
from src.portfolio_sim.config import (
    INITIAL_CAPITAL,
    KAMA_BUFFER,
    KAMA_PERIOD,
    LOOKBACK_PERIOD,
    TOP_N,
    VOLATILITY_LOOKBACK,
)
from src.portfolio_sim.data import fetch_etf_tickers, fetch_price_data
from src.portfolio_sim.engine import run_simulation
from src.portfolio_sim.models import SimulationResult
from src.portfolio_sim.params import StrategyParams
from src.portfolio_sim.reporting import (
    compute_drawdown_series,
    compute_metrics,
    compute_monthly_returns,
    compute_rolling_sharpe,
    compute_yearly_returns,
)

# ---------------------------------------------------------------------------
# Theme / CSS
# ---------------------------------------------------------------------------
CHART_TEMPLATE = "plotly_white"
COLOR_STRATEGY = "#2962FF"
COLOR_BENCHMARK = "#888888"
COLOR_POSITIVE = "#00c853"
COLOR_NEGATIVE = "#ff1744"
COLOR_DRAWDOWN = "#e74c3c"
COLOR_FRONTIER = "#FFD700"
COLOR_PORTFOLIO = "#00E676"


def _inject_css():
    st.markdown(
        """
        <style>
        /* --- Equal-height metric cards in a row --- */
        div[data-testid="stHorizontalBlock"] {
            align-items: stretch;
        }
        div[data-testid="stColumn"] > div:first-child {
            height: 100%;
        }
        /* --- Metric cards --- */
        .metric-card {
            background: #f5f5f5;
            border-radius: 8px;
            padding: 14px 18px;
            margin: 4px 0;
            height: 100%;
            box-sizing: border-box;
        }
        .metric-card .label {
            color: #666;
            font-size: 0.8rem;
            margin-bottom: 2px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .metric-card .value {
            font-size: 1.35rem;
            font-weight: 700;
        }
        .metric-card.positive { border-left: 4px solid #00c853; }
        .metric-card.positive .value { color: #00c853; }
        .metric-card.negative { border-left: 4px solid #ff1744; }
        .metric-card.negative .value { color: #ff1744; }
        .metric-card.neutral { border-left: 4px solid #888; }
        .metric-card.neutral .value { color: #333; }
        .section-divider {
            border-top: 1px solid #ddd;
            margin: 1.5rem 0 1rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _setup_logging():
    if "logging_configured" not in st.session_state:
        logging.basicConfig(level=logging.WARNING, format="%(message)s")
        structlog.configure(
            processors=[
                structlog.stdlib.add_log_level,
                structlog.dev.ConsoleRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING),
            logger_factory=structlog.stdlib.LoggerFactory(),
        )
        st.session_state.logging_configured = True



# ---------------------------------------------------------------------------
# Cached data helpers
# ---------------------------------------------------------------------------
def _fetch_etf_impl():
    return fetch_etf_tickers()


def _fetch_prices_impl(tickers_tuple: tuple, refresh: bool, cache_suffix: str = "", period: str = "5y"):
    return fetch_price_data(list(tickers_tuple), period=period, refresh=refresh, cache_suffix=cache_suffix)



# ---------------------------------------------------------------------------
# Metric card HTML
# ---------------------------------------------------------------------------
def _card_html(label: str, value: str, positive: bool | None = None) -> str:
    if positive is None:
        cls = "neutral"
    elif positive:
        cls = "positive"
    else:
        cls = "negative"
    return (
        f'<div class="metric-card {cls}">'
        f'<div class="label">{label}</div>'
        f'<div class="value">{value}</div>'
        f"</div>"
    )


def _render_metric_row(title: str, m: dict):
    st.markdown(f"**{title}**")
    cols = st.columns(7)
    cols[0].markdown(
        _card_html("Total Return", f"{m['total_return']:.1%}", m["total_return"] >= 0),
        unsafe_allow_html=True,
    )
    cols[1].markdown(
        _card_html("CAGR", f"{m['cagr']:.1%}", m["cagr"] >= 0),
        unsafe_allow_html=True,
    )
    cols[2].markdown(
        _card_html("Max Drawdown", f"{-m['max_drawdown']:.1%}", False),
        unsafe_allow_html=True,
    )
    cols[3].markdown(
        _card_html("Sharpe", f"{m['sharpe']:.2f}", m["sharpe"] >= 0),
        unsafe_allow_html=True,
    )
    cols[4].markdown(
        _card_html("Calmar", f"{m['calmar']:.2f}", m["calmar"] >= 0),
        unsafe_allow_html=True,
    )
    cols[5].markdown(
        _card_html("Ann. Volatility", f"{m['annualized_vol']:.1%}", None),
        unsafe_allow_html=True,
    )
    cols[6].markdown(
        _card_html("Win Rate", f"{m['win_rate']:.1%}", m["win_rate"] >= 0.5),
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------
def _render_comparison_table(strat_m: dict, spy_m: dict):
    rows = {
        "Total Return": (f"{strat_m['total_return']:.1%}", f"{spy_m['total_return']:.1%}"),
        "CAGR": (f"{strat_m['cagr']:.1%}", f"{spy_m['cagr']:.1%}"),
        "Max Drawdown": (f"{-strat_m['max_drawdown']:.1%}", f"{-spy_m['max_drawdown']:.1%}"),
        "Sharpe Ratio": (f"{strat_m['sharpe']:.2f}", f"{spy_m['sharpe']:.2f}"),
        "Calmar Ratio": (f"{strat_m['calmar']:.2f}", f"{spy_m['calmar']:.2f}"),
        "Ann. Volatility": (f"{strat_m['annualized_vol']:.1%}", f"{spy_m['annualized_vol']:.1%}"),
        "Win Rate": (f"{strat_m['win_rate']:.1%}", f"{spy_m['win_rate']:.1%}"),
        "Trading Days": (f"{strat_m['n_days']:,}", f"{spy_m['n_days']:,}"),
    }
    df = pd.DataFrame(rows, index=["Strategy", "S&P 500"]).T
    st.dataframe(df, width="stretch")


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def _base_layout(**kwargs) -> dict:
    defaults = dict(
        template=CHART_TEMPLATE,
        margin=dict(l=40, r=20, t=40, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    defaults.update(kwargs)
    return defaults


def plot_equity_curve(
    result: SimulationResult, log_scale: bool = False
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=result.equity.index,
            y=result.equity.values,
            mode="lines",
            name="KAMA Momentum",
            line=dict(color=COLOR_STRATEGY, width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=result.spy_equity.index,
            y=result.spy_equity.values,
            mode="lines",
            name="S&P 500 (Buy & Hold)",
            line=dict(color=COLOR_BENCHMARK, width=1.5, dash="dash"),
        )
    )

    yaxis_type = "log" if log_scale else "linear"
    fig.update_layout(
        **_base_layout(
            title="Equity Curve",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            height=480,
            yaxis_type=yaxis_type,
            xaxis=dict(rangeslider=dict(visible=True), type="date"),
        )
    )
    return fig


def plot_drawdowns(result: SimulationResult) -> go.Figure:
    dd_strat = compute_drawdown_series(result.equity) * 100
    dd_spy = compute_drawdown_series(result.spy_equity) * 100
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dd_strat.index,
            y=dd_strat.values,
            mode="lines",
            name="Strategy",
            line=dict(color=COLOR_DRAWDOWN, width=1.5),
            fill="tozeroy",
            fillcolor="rgba(231, 76, 60, 0.2)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dd_spy.index,
            y=dd_spy.values,
            mode="lines",
            name="S&P 500",
            line=dict(color=COLOR_BENCHMARK, width=1, dash="dash"),
            fill="tozeroy",
            fillcolor="rgba(136, 136, 136, 0.1)",
        )
    )
    fig.update_layout(
        **_base_layout(
            title="Drawdown",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            height=320,
        )
    )
    return fig


def plot_monthly_heatmap(equity: pd.Series, title: str = "Monthly Returns") -> go.Figure:
    table = compute_monthly_returns(equity)
    z = table.values * 100
    text = [[f"{v:.1f}%" if not np.isnan(v) else "" for v in row] for row in z]

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=table.columns.tolist(),
            y=[str(y) for y in table.index],
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=11),
            colorscale="RdYlGn",
            zmid=0,
            colorbar=dict(title="%", ticksuffix="%"),
        )
    )
    fig.update_layout(
        **_base_layout(title=title, height=max(250, len(table) * 40 + 80))
    )
    return fig


def plot_yearly_returns(
    strat_equity: pd.Series, spy_equity: pd.Series
) -> go.Figure:
    strat_yr = compute_yearly_returns(strat_equity) * 100
    spy_yr = compute_yearly_returns(spy_equity) * 100
    years = sorted(set(strat_yr.index) | set(spy_yr.index))

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[str(y) for y in years],
            y=[strat_yr.get(y, 0) for y in years],
            name="Strategy",
            marker_color=COLOR_STRATEGY,
        )
    )
    fig.add_trace(
        go.Bar(
            x=[str(y) for y in years],
            y=[spy_yr.get(y, 0) for y in years],
            name="S&P 500",
            marker_color=COLOR_BENCHMARK,
        )
    )
    fig.update_layout(
        **_base_layout(
            title="Annual Returns",
            yaxis_title="Return (%)",
            barmode="group",
            height=350,
        )
    )
    return fig


def plot_rolling_sharpe(
    strat_equity: pd.Series, spy_equity: pd.Series, window: int = 252
) -> go.Figure:
    strat_rs = compute_rolling_sharpe(strat_equity, window=window)
    spy_rs = compute_rolling_sharpe(spy_equity, window=window)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=strat_rs.index,
            y=strat_rs.values,
            mode="lines",
            name="Strategy",
            line=dict(color=COLOR_STRATEGY, width=1.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=spy_rs.index,
            y=spy_rs.values,
            mode="lines",
            name="S&P 500",
            line=dict(color=COLOR_BENCHMARK, width=1, dash="dash"),
        )
    )
    fig.add_hline(y=0, line_dash="dot", line_color="#555", line_width=1)
    fig.update_layout(
        **_base_layout(
            title=f"Rolling Sharpe Ratio ({window}-day)",
            xaxis_title="Date",
            yaxis_title="Sharpe Ratio",
            height=320,
        )
    )
    return fig


def plot_holdings_pie(
    result: SimulationResult, close_prices: pd.DataFrame
) -> go.Figure:
    last_date = result.equity.index[-1]
    last_holdings = result.holdings_history.loc[last_date]
    last_close = close_prices.loc[last_date]

    position_values = (last_holdings * last_close).dropna()
    position_values = position_values[position_values > 0].sort_values(ascending=False)
    cash = result.cash_history.iloc[-1]

    labels = list(position_values.index) + ["Cash"]
    values = list(position_values.values) + [cash]

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            textinfo="label+percent",
            textposition="outside",
            marker=dict(line=dict(color="#ffffff", width=1)),
        )
    )
    fig.update_layout(
        **_base_layout(title="Current Portfolio Allocation", height=450)
    )
    return fig


def plot_holdings_over_time(
    result: SimulationResult, close_prices: pd.DataFrame
) -> go.Figure:
    idx = result.holdings_history.index
    aligned_close = close_prices.reindex(index=idx, columns=result.holdings_history.columns)
    dollar_holdings = result.holdings_history * aligned_close

    # Only tickers that were ever held
    ever_held = dollar_holdings.columns[dollar_holdings.sum() > 0]
    dollar_holdings = dollar_holdings[ever_held]
    dollar_holdings["Cash"] = result.cash_history

    fig = go.Figure()
    for col in dollar_holdings.columns:
        fig.add_trace(
            go.Scatter(
                x=dollar_holdings.index,
                y=dollar_holdings[col].values,
                mode="lines",
                name=col,
                stackgroup="one",
                line=dict(width=0),
            )
        )
    fig.update_layout(
        **_base_layout(
            title="Portfolio Composition Over Time",
            xaxis_title="Date",
            yaxis_title="Value ($)",
            height=450,
            showlegend=False,
        )
    )
    return fig


def compute_efficient_frontier(
    close_prices: pd.DataFrame,
    n_points: int = 50,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Compute the efficient frontier for the given asset universe.

    The optimization uses arithmetic mean returns (correct for mean-variance
    optimization), but the returned values are converted to geometric returns
    (CAGR) for consistent display alongside realized portfolio metrics.

    Returns (frontier_vols, frontier_rets, asset_stats) where:
      - frontier_rets are CAGR-approximated returns (geometric ≈ arithmetic − σ²/2)
      - asset_stats is a DataFrame with columns ['ticker', 'ann_return', 'ann_vol']
        where 'ann_return' is the actual CAGR computed from price data.
    """
    daily_returns = close_prices.pct_change().dropna(how="all")
    valid_cols = [
        c for c in daily_returns.columns
        if daily_returns[c].dropna().shape[0] >= 252
    ]
    daily_returns = daily_returns[valid_cols].dropna()

    n_assets = len(valid_cols)
    if n_assets < 2:
        return np.array([]), np.array([]), pd.DataFrame()

    mean_daily = daily_returns.mean().values
    cov_daily = daily_returns.cov().values
    ann_returns = mean_daily * 252
    ann_cov = cov_daily * 252
    # Small regularization for numerical stability
    ann_cov += np.eye(n_assets) * 1e-8

    ann_vols = np.sqrt(np.diag(ann_cov))

    # Compute CAGR for each individual asset from actual price data
    # (more accurate than the arithmetic-to-geometric approximation).
    asset_cagrs = []
    for col in valid_cols:
        col_prices = close_prices[col].dropna()
        if len(col_prices) >= 2:
            asset_cagr = (
                (col_prices.iloc[-1] / col_prices.iloc[0])
                ** (252 / len(col_prices))
                - 1
            )
        else:
            asset_cagr = 0.0
        asset_cagrs.append(asset_cagr)

    asset_stats = pd.DataFrame({
        "ticker": valid_cols,
        "ann_return": asset_cagrs,
        "ann_vol": ann_vols,
    })

    def port_vol(w):
        return np.sqrt(w @ ann_cov @ w)

    def port_ret(w):
        return w @ ann_returns

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = tuple((0.0, 1.0) for _ in range(n_assets))
    w0 = np.ones(n_assets) / n_assets

    # Min variance portfolio
    res_min = sco.minimize(port_vol, w0, method="SLSQP",
                           bounds=bounds, constraints=constraints)
    min_ret = port_ret(res_min.x)
    max_ret = float(ann_returns.max())

    target_returns = np.linspace(min_ret, max_ret, n_points)
    frontier_vols = []
    frontier_rets = []

    for target in target_returns:
        cons = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, t=target: port_ret(w) - t},
        ]
        res = sco.minimize(port_vol, w0, method="SLSQP",
                           bounds=bounds, constraints=cons,
                           options={"ftol": 1e-9, "maxiter": 1000})
        if res.success:
            frontier_vols.append(port_vol(res.x))
            frontier_rets.append(port_ret(res.x))

    frontier_vols_arr = np.array(frontier_vols)
    frontier_rets_arr = np.array(frontier_rets)

    # Convert frontier returns from arithmetic to geometric (CAGR).
    # Standard approximation: geometric ≈ arithmetic − σ²/2
    frontier_rets_geo = frontier_rets_arr - 0.5 * frontier_vols_arr ** 2

    return frontier_vols_arr, frontier_rets_geo, asset_stats


def plot_risk_return_scatter(
    close_prices: pd.DataFrame,
    result: SimulationResult | None = None,
) -> go.Figure:
    """Plot risk-return scatter of individual assets with efficient frontier."""
    asset_cols = [c for c in close_prices.columns if c != "SPY"]
    asset_prices = close_prices[asset_cols]

    frontier_vols, frontier_rets, asset_stats = compute_efficient_frontier(
        asset_prices
    )

    fig = go.Figure()

    # Individual asset scatter
    if not asset_stats.empty:
        fig.add_trace(
            go.Scatter(
                x=asset_stats["ann_vol"].values * 100,
                y=asset_stats["ann_return"].values * 100,
                mode="markers",
                name="Assets",
                marker=dict(
                    size=6,
                    color=asset_stats["ann_return"].values * 100,
                    colorscale="RdYlGn",
                    showscale=True,
                    colorbar=dict(title="CAGR %", ticksuffix="%"),
                    opacity=0.7,
                    line=dict(width=0.5, color="#333"),
                ),
                text=asset_stats["ticker"].values,
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Volatility: %{x:.1f}%<br>"
                    "CAGR: %{y:.1f}%<br>"
                    "<extra></extra>"
                ),
            )
        )

    # Efficient frontier curve
    if len(frontier_vols) > 0:
        fig.add_trace(
            go.Scatter(
                x=frontier_vols * 100,
                y=frontier_rets * 100,
                mode="lines",
                name="Efficient Frontier",
                line=dict(color=COLOR_FRONTIER, width=3),
                hovertemplate=(
                    "Volatility: %{x:.1f}%<br>"
                    "CAGR: %{y:.1f}%<br>"
                    "<extra>Efficient Frontier</extra>"
                ),
            )
        )

    # Current portfolio position
    if result is not None:
        strat_m = compute_metrics(result.equity)
        fig.add_trace(
            go.Scatter(
                x=[strat_m["annualized_vol"] * 100],
                y=[strat_m["cagr"] * 100],
                mode="markers+text",
                name="Portfolio",
                marker=dict(
                    size=16, color=COLOR_PORTFOLIO, symbol="star",
                    line=dict(width=2, color="#fff"),
                ),
                text=["Portfolio"],
                textposition="top center",
                textfont=dict(color=COLOR_PORTFOLIO, size=12),
                hovertemplate=(
                    "<b>KAMA Portfolio</b><br>"
                    "Volatility: %{x:.1f}%<br>"
                    "CAGR: %{y:.1f}%<br>"
                    "<extra></extra>"
                ),
            )
        )
        spy_m = compute_metrics(result.spy_equity)
        fig.add_trace(
            go.Scatter(
                x=[spy_m["annualized_vol"] * 100],
                y=[spy_m["cagr"] * 100],
                mode="markers+text",
                name="S&P 500",
                marker=dict(
                    size=14, color=COLOR_BENCHMARK, symbol="diamond",
                    line=dict(width=2, color="#fff"),
                ),
                text=["SPY"],
                textposition="top center",
                textfont=dict(color=COLOR_BENCHMARK, size=11),
                hovertemplate=(
                    "<b>S&P 500</b><br>"
                    "Volatility: %{x:.1f}%<br>"
                    "CAGR: %{y:.1f}%<br>"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        **_base_layout(
            title="Risk-Return Distribution & Efficient Frontier",
            xaxis_title="Annualized Volatility (%)",
            yaxis_title="CAGR (%)",
            height=550,
        )
    )
    return fig


def _render_trade_log(result: SimulationResult):
    if not result.trade_log:
        st.info("No trades recorded.")
        return

    df = pd.DataFrame(result.trade_log)
    df["trade_value"] = df["shares"] * df["price"]
    df["date"] = pd.to_datetime(df["date"]).dt.date

    col1, col2, col3 = st.columns(3)
    buys = df[df["action"] == "buy"]
    sells = df[df["action"] == "sell"]
    col1.metric("Total Trades", len(df))
    col2.metric("Buys", len(buys))
    col3.metric("Sells", len(sells))

    st.dataframe(
        df[["date", "ticker", "action", "shares", "price", "trade_value"]].rename(
            columns={
                "date": "Date",
                "ticker": "Ticker",
                "action": "Action",
                "shares": "Shares",
                "price": "Price",
                "trade_value": "Trade Value",
            }
        ),
        width="stretch",
        height=400,
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def _render_sidebar() -> dict:
    with st.sidebar:
        st.title("KAMA Momentum")

        # Перемещаем кнопку запуска наверх для удобства
        run_clicked = st.button(
            "🚀 Запустить бэктест", 
            type="primary", 
            width="stretch",
        )

        is_etf_mode = True
        universe_mode = "ETF Cross-Asset"

        # --- Группа 1: Данные и Оптимизация ---
        with st.expander("📊 Данные и Оптимизация", expanded=True):
            data_years = st.slider(
                "Период данных (лет)",
                min_value=3, max_value=10, value=3,
                help="Количество лет исторических данных для загрузки"
            )

            refresh = st.checkbox("Обновить кэш данных", value=False)

            optimize_mode = st.selectbox(
                "Режим оптимизации",
                ["None", "Sensitivity Analysis", "Max Profit (TPE)",
                 "Walk-Forward"],
                help="Метод поиска параметров перед бэктестом"
            )

            opt_n_trials = 50
            opt_max_dd = 0.60
            opt_oos_days = 126
            opt_min_is_days = 378

            if optimize_mode != "None":
                opt_n_trials = st.slider(
                    "Количество итераций Optuna", min_value=20, max_value=200,
                    value=50, step=10,
                )

            if optimize_mode == "Max Profit (TPE)":
                opt_max_dd = st.slider(
                    "Лимит просадки (%)", min_value=10, max_value=90,
                    value=60, step=5,
                    help="Отклонять комбинации с просадкой выше этого лимита.",
                ) / 100.0

            if optimize_mode == "Walk-Forward":
                opt_oos_days = st.slider(
                    "OOS окно (дни)", min_value=63, max_value=252,
                    value=126, step=21,
                    help="Окно проверки вне выборки на каждом шаге (~6 месяцев = 126).",
                )
                opt_min_is_days = st.slider(
                    "Мин. IS окно (дни)", min_value=252, max_value=756,
                    value=378, step=63,
                    help="Минимальное окно обучения внутри выборки (~1.5 года = 378).",
                )

        # --- Группа 2: Параметры стратегии ---
        with st.expander("⚙️ Параметры стратегии", expanded=True):
            initial_capital = st.number_input(
                "Начальный капитал ($)", 
                min_value=1_000.0, max_value=10_000_000.0,
                value=float(INITIAL_CAPITAL), step=1_000.0,
                help="Стартовая стоимость портфеля"
            )

            opt = st.session_state.get("optimized_params")

            _default_top_n = opt.top_n if opt else TOP_N
            top_n = st.slider(
                "Кол-во активов (Top N)",
                min_value=3, max_value=20 if is_etf_mode else 50,
                value=min(_default_top_n, 20) if is_etf_mode else _default_top_n,
                help="Количество позиций, удерживаемых одновременно"
            )

            _default_kama = opt.kama_period if opt else KAMA_PERIOD
            kama_period = st.slider(
                "Период KAMA", min_value=5, max_value=50, value=_default_kama,
                help="Окно адаптивной скользящей средней (торговые дни)"
            )

            _default_lookback = opt.lookback_period if opt else LOOKBACK_PERIOD
            lookback_period = st.slider(
                "Окно моментума", min_value=20, max_value=252, value=_default_lookback,
                help="Окно оценки моментума (торговые дни)"
            )

            _default_buffer = opt.kama_buffer if opt else KAMA_BUFFER
            kama_buffer = st.slider(
                "Буфер KAMA", min_value=0.0, max_value=0.05,
                value=float(_default_buffer), step=0.001, format="%.3f",
                help="Порог гистерезиса для переключения режимов"
            )

            use_risk_adjusted = st.toggle(
                "Риск-адаптированный моментум",
                value=True if is_etf_mode else False,
                help="Ранжирование по доходность × ER² (Efficiency Ratio). "
                     "Жёстко штрафует хаотичное движение, "
                     "предпочитает плавные восходящие тренды.",
            )

            kama_slope_filter = st.toggle(
                "KAMA slope gate",
                value=False,
                help="Требовать, чтобы KAMA была направлена вверх для входа. "
                     "Дополнительная фильтрация: не покупать активы с плоской KAMA.",
            )

        # --- Группа 3: Расширенные настройки ---
        with st.expander("⚙️ Расширенные настройки", expanded=False):
            sizing_options = ["Equal Weight", "Risk Parity (Inverse Vol)"]
            sizing_choice = st.radio(
                "Метод определения весов",
                options=sizing_options,
                index=1 if is_etf_mode else 0,
                help="Risk Parity: больше веса низковолатильным активам (облигации) и меньше высоковолатильным (акции).",
            )
            sizing_mode = "risk_parity" if sizing_choice == sizing_options[1] else "equal_weight"
            vol_lookback = VOLATILITY_LOOKBACK
            if sizing_mode == "risk_parity":
                vol_lookback = st.slider(
                    "Окно волатильности (дни)", min_value=10, max_value=60,
                    value=VOLATILITY_LOOKBACK,
                    help="Окно для расчета весов на основе обратной волатильности"
                )

            max_weight = st.slider(
                "Макс. вес позиции (%)",
                min_value=5, max_value=50, value=15, step=1,
                help="Ограничение веса любого отдельного актива в портфеле"
            ) / 100.0

    return {
        "universe_mode": universe_mode,
        "is_etf_mode": is_etf_mode,
        "data_years": data_years,
        "refresh": refresh,
        "optimize_mode": optimize_mode,
        "opt_n_trials": opt_n_trials,
        "opt_max_dd": opt_max_dd,
        "opt_oos_days": opt_oos_days,
        "opt_min_is_days": opt_min_is_days,
        "initial_capital": float(initial_capital),
        "top_n": top_n,
        "kama_period": kama_period,
        "lookback_period": lookback_period,
        "kama_buffer": kama_buffer,
        "use_risk_adjusted": use_risk_adjusted,
        "kama_slope_filter": kama_slope_filter,
        "sizing_mode": sizing_mode,
        "vol_lookback": vol_lookback,
        "max_weight": max_weight,
        "run_clicked": run_clicked,
    }


# ---------------------------------------------------------------------------
# Optimization results display
# ---------------------------------------------------------------------------
def _render_optimization_results() -> None:
    """Show optimization details if available in session state."""
    detail = st.session_state.get("opt_detail")
    if detail is None:
        return

    kind, data = detail

    if kind == "sensitivity":
        with st.expander("Sensitivity Analysis Results", expanded=False):
            # Robustness scores
            st.subheader("Robustness Scores")
            scores = data.robustness_scores
            cols = st.columns(len(scores))
            for col, (name, score) in zip(cols, scores.items()):
                verdict = "Robust" if score >= 0.8 else "Moderate" if score >= 0.5 else "Sensitive"
                col.metric(name, f"{score:.2f}", verdict)

            # Top combos
            valid = data.grid_results[data.grid_results["objective"] > -999.0]
            if not valid.empty:
                st.subheader("Top 10 Combinations")
                top = valid.nlargest(10, "objective")[
                    ["kama_period", "lookback_period", "kama_buffer", "top_n",
                     "objective", "cagr", "max_drawdown", "sharpe"]
                ].reset_index(drop=True)
                top.index += 1
                fmt = {
                    "cagr": "{:.2%}",
                    "max_drawdown": "{:.2%}",
                    "objective": "{:.4f}",
                    "sharpe": "{:.2f}",
                    "kama_buffer": "{:.3f}",
                }
                st.dataframe(top.style.format(fmt), width="stretch")

    elif kind == "max_profit_tpe":
        with st.expander("Max Profit (TPE) Results", expanded=False):
            grid = data.grid_results
            valid = grid[grid["objective_cagr"] > -999.0]
            st.caption(
                f"{len(grid)} trials, {len(valid)} valid "
                f"({len(valid) / max(len(grid), 1):.0%})"
            )
            if not valid.empty:
                st.subheader("Top 10 by CAGR")
                display_cols = ["kama_period", "lookback_period", "kama_buffer",
                                "top_n", "cagr", "max_drawdown", "sharpe"]
                existing = [c for c in display_cols if c in valid.columns]
                top = valid.nlargest(10, "cagr")[existing].reset_index(drop=True)
                top.index += 1
                fmt = {"cagr": "{:.2%}", "max_drawdown": "{:.2%}",
                       "sharpe": "{:.2f}", "kama_buffer": "{:.3f}"}
                st.dataframe(top.style.format(
                    {k: v for k, v in fmt.items() if k in existing}
                ), width="stretch")

    elif kind == "wfo":
        with st.expander("Walk-Forward Optimization Results", expanded=False):
            wfo = data
            # Per-step table
            step_rows = []
            for step in wfo.steps:
                step_rows.append({
                    "Step": step.step_index + 1,
                    "IS Period": f"{step.is_start.date()} .. {step.is_end.date()}",
                    "OOS Period": f"{step.oos_start.date()} .. {step.oos_end.date()}",
                    "IS CAGR": step.is_metrics.get("cagr", 0),
                    "OOS CAGR": step.oos_metrics.get("cagr", 0),
                    "OOS MaxDD": step.oos_metrics.get("max_drawdown", 0),
                    "OOS Sharpe": step.oos_metrics.get("sharpe", 0),
                })
            steps_df = pd.DataFrame(step_rows)
            fmt = {"IS CAGR": "{:.2%}", "OOS CAGR": "{:.2%}",
                   "OOS MaxDD": "{:.2%}", "OOS Sharpe": "{:.2f}"}
            st.dataframe(steps_df.style.format(fmt), width="stretch")

            # Degradation
            import numpy as np
            is_cagrs = [s.is_metrics.get("cagr", 0) for s in wfo.steps]
            oos_cagrs = [s.oos_metrics.get("cagr", 0) for s in wfo.steps]
            avg_is = np.mean(is_cagrs)
            avg_oos = np.mean(oos_cagrs)
            if avg_is > 0:
                degradation = 1.0 - avg_oos / avg_is
                verdict = "Acceptable" if degradation <= 0.5 else "High — possible overfitting"
                st.metric("IS/OOS Degradation", f"{degradation:.1%}", verdict)
            else:
                st.metric("IS/OOS Degradation", "N/A")

            # Stitched OOS equity chart
            st.subheader("Stitched Out-of-Sample Equity")
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=wfo.stitched_equity.index,
                y=wfo.stitched_equity.values,
                name="Strategy (OOS)",
                line=dict(width=2),
            ))
            if wfo.stitched_spy_equity is not None and not wfo.stitched_spy_equity.empty:
                fig.add_trace(go.Scatter(
                    x=wfo.stitched_spy_equity.index,
                    y=wfo.stitched_spy_equity.values,
                    name="SPY (OOS)",
                    line=dict(width=1, dash="dash"),
                ))
            fig.update_layout(
                yaxis_title="Portfolio Value ($)",
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig, width="stretch")

            # Recommended params
            fp = wfo.final_params
            st.info(
                f"Recommended live params: "
                f"KAMA={fp.kama_period}, Lookback={fp.lookback_period}, "
                f"Buffer={fp.kama_buffer}, Top N={fp.top_n}"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    _setup_logging()

    cached_fetch_etf = st.cache_data(show_spinner="Fetching ETF tickers...")(_fetch_etf_impl)
    cached_fetch_prices = st.cache_data(show_spinner="Downloading price data...")(_fetch_prices_impl)

    _inject_css()
    sidebar = _render_sidebar()

    st.title("KAMA Momentum Strategy")

    st.caption(
        "Long/Cash only • Cross-asset ETFs • "
        + ("Risk Parity" if sidebar["sizing_mode"] == "risk_parity" else "Equal Weight")
        + " • Daily KAMA review"
    )

    if sidebar["run_clicked"]:
        universe = cached_fetch_etf()
        period = f"{sidebar['data_years']}y"
        cache_suffix = f"_etf_{period}"

        close_prices, open_prices = cached_fetch_prices(
            tuple(sorted(universe)), sidebar["refresh"],
            cache_suffix=cache_suffix, period=period,
        )

        min_days = int(sidebar["data_years"] * 252 * 0.6)
        valid = filter_valid_tickers(close_prices, min_days)

        # Base param kwargs shared across all branches
        common_kwargs = dict(
            use_risk_adjusted=sidebar["use_risk_adjusted"],
            kama_slope_filter=sidebar["kama_slope_filter"],
            sizing_mode=sidebar["sizing_mode"],
            volatility_lookback=sidebar["vol_lookback"],
            max_weight=sidebar["max_weight"],
        )

        # --- Optimization dispatch ---
        best = None
        opt_mode = sidebar["optimize_mode"]
        n_trials = sidebar["opt_n_trials"]

        if opt_mode == "Sensitivity Analysis":
            with st.spinner(f"Sensitivity analysis ({n_trials} trials)..."):
                from src.portfolio_sim.optimizer import (
                    find_best_params, run_sensitivity,
                )
                sens_result = run_sensitivity(
                    close_prices, open_prices, valid,
                    sidebar["initial_capital"],
                    n_trials=n_trials,
                    n_workers=-1,
                )
                best = find_best_params(sens_result)
                st.session_state["opt_detail"] = ("sensitivity", sens_result)

        elif opt_mode == "Max Profit (TPE)":
            fixed = {
                "use_risk_adjusted": True,
                "sizing_mode": "risk_parity",
            }
            from src.portfolio_sim.max_profit import run_max_profit_search
            from src.portfolio_sim.params import StrategyParams as _SP

            base = _SP(
                use_risk_adjusted=True,
                sizing_mode="risk_parity",
            )

            with st.spinner(f"Max-profit TPE search ({n_trials} trials)..."):
                mp_result = run_max_profit_search(
                    close_prices, open_prices, valid,
                    sidebar["initial_capital"],
                    universe="etf",
                    default_params=base,
                    fixed_params=fixed,
                    n_trials=n_trials,
                    max_dd_limit=sidebar["opt_max_dd"],
                )
                grid = mp_result.grid_results
                top = grid[grid["objective_cagr"] > -999.0]
                if not top.empty:
                    b = top.nlargest(1, "cagr").iloc[0]
                    best = StrategyParams(
                        kama_period=int(b["kama_period"]),
                        lookback_period=int(b["lookback_period"]),
                        top_n=int(b["top_n"]),
                        kama_buffer=float(b["kama_buffer"]),
                    )
                st.session_state["opt_detail"] = ("max_profit_tpe", mp_result)

            # Override diversification/sizing with max-profit fixed params
            if best is not None:
                common_kwargs.update(
                    use_risk_adjusted=True,
                    sizing_mode="risk_parity",
                )

        elif opt_mode == "Walk-Forward":
            from src.portfolio_sim.walk_forward import run_walk_forward

            with st.spinner(
                f"Walk-forward optimization ({n_trials} trials/step, "
                f"OOS={sidebar['opt_oos_days']}d)..."
            ):
                wfo_result = run_walk_forward(
                    close_prices, open_prices, valid,
                    sidebar["initial_capital"],
                    n_trials_per_step=n_trials,
                    min_is_days=sidebar["opt_min_is_days"],
                    oos_days=sidebar["opt_oos_days"],
                )
                best = wfo_result.final_params
                st.session_state["opt_detail"] = ("wfo", wfo_result)

        if best is not None:
            st.session_state["optimized_params"] = best
            st.toast(
                f"Optimal: KAMA={best.kama_period}, "
                f"Lookback={best.lookback_period}, "
                f"Buffer={best.kama_buffer}, "
                f"Top N={best.top_n}",
                icon="✅",
            )
            common_kwargs.update(
                kama_period=best.kama_period,
                lookback_period=best.lookback_period,
                top_n=best.top_n,
                kama_buffer=best.kama_buffer,
            )
        elif opt_mode != "None":
            st.warning("Optimization found no valid combinations. Using manual parameters.")
            common_kwargs.update(
                kama_period=sidebar["kama_period"],
                lookback_period=sidebar["lookback_period"],
                top_n=sidebar["top_n"],
                kama_buffer=sidebar["kama_buffer"],
            )
        else:
            common_kwargs.update(
                kama_period=sidebar["kama_period"],
                lookback_period=sidebar["lookback_period"],
                top_n=sidebar["top_n"],
                kama_buffer=sidebar["kama_buffer"],
            )

        params = StrategyParams(**common_kwargs)

        with st.spinner(f"Running simulation on {len(valid)} tickers..."):
            result = run_simulation(
                close_prices, open_prices, valid,
                sidebar["initial_capital"], params=params,
            )

        st.session_state["result"] = result
        st.session_state["close_prices"] = close_prices

    if "result" not in st.session_state:
        st.info("Click **Run Backtest** in the sidebar to start.")
        return

    result: SimulationResult = st.session_state["result"]
    close_prices: pd.DataFrame = st.session_state["close_prices"]

    strat_m = compute_metrics(result.equity)
    spy_m = compute_metrics(result.spy_equity)

    # --- Optimization results expander ---
    _render_optimization_results()

    # --- Date range info ---
    st.markdown(
        f"**Simulation period:** {result.equity.index[0].strftime('%Y-%m-%d')} "
        f"to {result.equity.index[-1].strftime('%Y-%m-%d')} "
        f"({strat_m['n_days']:,} trading days)"
    )

    # --- Section 1: Metric Cards ---
    _render_metric_row("Strategy", strat_m)
    _render_metric_row("S&P 500 Benchmark", spy_m)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # --- Section 2: Comparison Table ---
    with st.expander("Detailed Comparison", expanded=False):
        _render_comparison_table(strat_m, spy_m)

    # --- Section 3: Equity Curve ---
    log_scale = st.checkbox("Log scale", value=False)
    st.plotly_chart(
        plot_equity_curve(result, log_scale=log_scale), width="stretch"
    )

    # --- Section 4: Drawdown ---
    st.plotly_chart(plot_drawdowns(result), width="stretch")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # --- Section 5: Monthly Heatmap + Yearly Bar ---
    col_left, col_right = st.columns(2)
    with col_left:
        st.plotly_chart(
            plot_monthly_heatmap(result.equity, "Strategy Monthly Returns"),
            width="stretch",
        )
    with col_right:
        st.plotly_chart(
            plot_yearly_returns(result.equity, result.spy_equity),
            width="stretch",
        )

    # --- Section 6: Rolling Sharpe ---
    st.plotly_chart(
        plot_rolling_sharpe(result.equity, result.spy_equity),
        width="stretch",
    )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # --- Section 6b: Risk-Return Scatter & Efficient Frontier ---
    with st.spinner("Computing efficient frontier..."):
        st.plotly_chart(
            plot_risk_return_scatter(close_prices, result=result),
            width="stretch",
        )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # --- Section 7: Portfolio Composition ---
    tab_pie, tab_area = st.tabs(["Current Holdings", "Holdings Over Time"])
    with tab_pie:
        st.plotly_chart(
            plot_holdings_pie(result, close_prices), width="stretch"
        )
    with tab_area:
        st.plotly_chart(
            plot_holdings_over_time(result, close_prices),
            width="stretch",
        )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # --- Section 8: Trade Log ---
    with st.expander("Trade Log", expanded=False):
        _render_trade_log(result)


if __name__ == "__main__":
    st.set_page_config(
        page_title="KAMA Momentum Strategy",
        page_icon="\U0001F4C8",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    main()
