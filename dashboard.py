"""Streamlit dashboard for R² Momentum strategy (Clenow-style).

Run with: streamlit run dashboard.py
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.optimize as sco
import streamlit as st
import structlog

from scripts.compare_methods import run_backtest
from src.portfolio_sim.cli_utils import filter_valid_tickers
from src.portfolio_sim.config import (
    ATR_PERIOD,
    DEFAULT_N_TRIALS,
    GAP_THRESHOLD,
    INITIAL_CAPITAL,
    KAMA_BUFFER,
    KAMA_PERIOD,
    KAMA_SPY_PERIOD,
    OOS_DAYS,
    R2_LOOKBACK,
    REBAL_PERIOD_WEEKS,
    RISK_FACTOR,
    TOP_N,
)
from src.portfolio_sim.data import fetch_etf_tickers, fetch_price_data
from src.portfolio_sim.reporting import (
    compute_drawdown_series,
    compute_metrics,
    compute_monthly_returns,
    compute_rolling_sharpe,
    compute_yearly_returns,
)


# ---------------------------------------------------------------------------
# Result container (duck-typed replacement for old SimulationResult)
# ---------------------------------------------------------------------------
@dataclass
class BacktestResult:
    """Container for R² Momentum backtest output."""

    equity: pd.Series
    spy_equity: pd.Series
    holdings_history: pd.DataFrame
    cash_history: pd.Series
    trade_log: list[dict] = field(default_factory=list)


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
    result: BacktestResult, log_scale: bool = False
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=result.equity.index,
            y=result.equity.values,
            mode="lines",
            name="R\u00b2 Momentum",
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


def plot_drawdowns(result: BacktestResult) -> go.Figure:
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
    result: BacktestResult, close_prices: pd.DataFrame
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
    result: BacktestResult, close_prices: pd.DataFrame
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
      - frontier_rets are CAGR-approximated returns (geometric ~ arithmetic - sigma^2/2)
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
    frontier_rets_geo = frontier_rets_arr - 0.5 * frontier_vols_arr ** 2

    return frontier_vols_arr, frontier_rets_geo, asset_stats


def plot_risk_return_scatter(
    close_prices: pd.DataFrame,
    result: BacktestResult | None = None,
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
                    "<b>R\u00b2 Momentum Portfolio</b><br>"
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


def _render_trade_log(result: BacktestResult):
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
        st.title("R\u00b2 Momentum")

        run_clicked = st.button(
            "\u0417\u0430\u043f\u0443\u0441\u0442\u0438\u0442\u044c \u0431\u044d\u043a\u0442\u0435\u0441\u0442",
            type="primary",
            width="stretch",
        )

        # --- Group 1: Data & Optimization ---
        with st.expander("\u0414\u0430\u043d\u043d\u044b\u0435 \u0438 \u041e\u043f\u0442\u0438\u043c\u0438\u0437\u0430\u0446\u0438\u044f", expanded=True):
            data_years = st.slider(
                "\u041f\u0435\u0440\u0438\u043e\u0434 \u0434\u0430\u043d\u043d\u044b\u0445 (\u043b\u0435\u0442)",
                min_value=3, max_value=5, value=3,
                help="\u041a\u043e\u043b\u0438\u0447\u0435\u0441\u0442\u0432\u043e \u043b\u0435\u0442 \u0438\u0441\u0442\u043e\u0440\u0438\u0447\u0435\u0441\u043a\u0438\u0445 \u0434\u0430\u043d\u043d\u044b\u0445 \u0434\u043b\u044f \u0437\u0430\u0433\u0440\u0443\u0437\u043a\u0438",
            )

            refresh = st.checkbox("\u041e\u0431\u043d\u043e\u0432\u0438\u0442\u044c \u043a\u044d\u0448 \u0434\u0430\u043d\u043d\u044b\u0445", value=False)

            optimize_mode = st.selectbox(
                "\u0420\u0435\u0436\u0438\u043c \u043e\u043f\u0442\u0438\u043c\u0438\u0437\u0430\u0446\u0438\u0438",
                ["None", "Walk-Forward"],
                help="Walk-Forward Optimization: \u043e\u043f\u0442\u0438\u043c\u0438\u0437\u0430\u0446\u0438\u044f \u043d\u0430 IS-\u043e\u043a\u043d\u0435, \u0432\u0430\u043b\u0438\u0434\u0430\u0446\u0438\u044f \u043d\u0430 OOS",
            )

            opt_n_trials = DEFAULT_N_TRIALS
            opt_oos_days = OOS_DAYS
            opt_min_is_days = 126

            if optimize_mode == "Walk-Forward":
                opt_n_trials = st.slider(
                    "\u0418\u0442\u0435\u0440\u0430\u0446\u0438\u0438 Optuna (\u043d\u0430 \u0448\u0430\u0433)", min_value=20, max_value=500,
                    value=DEFAULT_N_TRIALS, step=10,
                )
                opt_oos_days = st.slider(
                    "OOS \u043e\u043a\u043d\u043e (\u0434\u043d\u0438)", min_value=10, max_value=63,
                    value=OOS_DAYS, step=7,
                    help="\u041e\u043a\u043d\u043e \u043f\u0440\u043e\u0432\u0435\u0440\u043a\u0438 \u0432\u043d\u0435 \u0432\u044b\u0431\u043e\u0440\u043a\u0438 \u043d\u0430 \u043a\u0430\u0436\u0434\u043e\u043c \u0448\u0430\u0433\u0435 WFO.",
                )
                opt_min_is_days = st.slider(
                    "IS \u043e\u043a\u043d\u043e (\u0434\u043d\u0438)", min_value=63, max_value=252,
                    value=126, step=21,
                    help="\u041c\u0438\u043d\u0438\u043c\u0430\u043b\u044c\u043d\u043e\u0435 \u043e\u043a\u043d\u043e \u043e\u043f\u0442\u0438\u043c\u0438\u0437\u0430\u0446\u0438\u0438 (in-sample).",
                )

        # --- Group 2: Strategy Parameters ---
        with st.expander("\u041f\u0430\u0440\u0430\u043c\u0435\u0442\u0440\u044b \u0441\u0442\u0440\u0430\u0442\u0435\u0433\u0438\u0438", expanded=True):
            initial_capital = st.number_input(
                "\u041d\u0430\u0447\u0430\u043b\u044c\u043d\u044b\u0439 \u043a\u0430\u043f\u0438\u0442\u0430\u043b ($)",
                min_value=1_000.0, max_value=10_000_000.0,
                value=float(INITIAL_CAPITAL), step=1_000.0,
                help="\u0421\u0442\u0430\u0440\u0442\u043e\u0432\u0430\u044f \u0441\u0442\u043e\u0438\u043c\u043e\u0441\u0442\u044c \u043f\u043e\u0440\u0442\u0444\u0435\u043b\u044f",
            )

            opt = st.session_state.get("optimized_params")

            _default_r2_lb = opt.r2_lookback if opt else R2_LOOKBACK
            r2_lookback = st.slider(
                "R\u00b2 Lookback (\u0434\u043d\u0438)", min_value=20, max_value=120, step=20,
                value=_default_r2_lb,
                help="\u041e\u043a\u043d\u043e OLS-\u0440\u0435\u0433\u0440\u0435\u0441\u0441\u0438\u0438 \u0434\u043b\u044f \u0441\u043a\u043e\u0440\u0438\u043d\u0433\u0430 \u043c\u043e\u043c\u0435\u043d\u0442\u0443\u043c\u0430 (Clenow)",
            )

            _default_top_n = opt.top_n if opt else TOP_N
            top_n = st.slider(
                "\u041a\u043e\u043b-\u0432\u043e \u0430\u043a\u0442\u0438\u0432\u043e\u0432 (Top N)",
                min_value=5, max_value=25, step=5,
                value=min(_default_top_n, 25),
                help="\u041a\u043e\u043b\u0438\u0447\u0435\u0441\u0442\u0432\u043e \u043f\u043e\u0437\u0438\u0446\u0438\u0439, \u0443\u0434\u0435\u0440\u0436\u0438\u0432\u0430\u0435\u043c\u044b\u0445 \u043e\u0434\u043d\u043e\u0432\u0440\u0435\u043c\u0435\u043d\u043d\u043e",
            )

            _default_kama = opt.kama_asset_period if opt else KAMA_PERIOD
            kama_asset_period = st.slider(
                "\u041f\u0435\u0440\u0438\u043e\u0434 KAMA (\u0430\u043a\u0442\u0438\u0432)", min_value=10, max_value=50, value=_default_kama,
                help="KAMA \u0434\u043b\u044f \u0438\u043d\u0434\u0438\u0432\u0438\u0434\u0443\u0430\u043b\u044c\u043d\u043e\u0433\u043e \u0442\u0440\u0435\u043d\u0434-\u0444\u0438\u043b\u044c\u0442\u0440\u0430 (\u0442\u043e\u0440\u0433\u043e\u0432\u044b\u0435 \u0434\u043d\u0438)",
            )

            _default_kama_spy = opt.kama_spy_period if opt else KAMA_SPY_PERIOD
            kama_spy_period = st.slider(
                "\u041f\u0435\u0440\u0438\u043e\u0434 KAMA (SPY)", min_value=10, max_value=60,
                value=_default_kama_spy,
                help="\u041f\u0435\u0440\u0438\u043e\u0434 KAMA \u0434\u043b\u044f \u0440\u0435\u0436\u0438\u043c\u043d\u043e\u0433\u043e \u0444\u0438\u043b\u044c\u0442\u0440\u0430 SPY (\u0442\u043e\u0440\u0433\u043e\u0432\u044b\u0435 \u0434\u043d\u0438)",
            )

            _default_buffer = opt.kama_buffer if opt else KAMA_BUFFER
            kama_buffer = st.slider(
                "\u0411\u0443\u0444\u0435\u0440 KAMA", min_value=0.005, max_value=0.05,
                value=float(_default_buffer), step=0.005, format="%.3f",
                help="\u041f\u043e\u0440\u043e\u0433 \u0433\u0438\u0441\u0442\u0435\u0440\u0435\u0437\u0438\u0441\u0430 \u0434\u043b\u044f \u043f\u0435\u0440\u0435\u043a\u043b\u044e\u0447\u0435\u043d\u0438\u044f \u0440\u0435\u0436\u0438\u043c\u043e\u0432",
            )

            _default_rebal = opt.rebal_period_weeks if opt else REBAL_PERIOD_WEEKS
            rebal_period_weeks = st.slider(
                "\u041f\u0435\u0440\u0438\u043e\u0434 \u0440\u0435\u0431\u0430\u043b\u0430\u043d\u0441\u0438\u0440\u043e\u0432\u043a\u0438 (\u043d\u0435\u0434.)",
                min_value=1, max_value=6, step=1,
                value=_default_rebal,
                help="\u041a\u0430\u043a \u0447\u0430\u0441\u0442\u043e \u043f\u0435\u0440\u0435\u0441\u043c\u0430\u0442\u0440\u0438\u0432\u0430\u0435\u043c \u043f\u043e\u0440\u0442\u0444\u0435\u043b\u044c (lazy-hold \u0440\u0435\u0431\u0430\u043b\u0430\u043d\u0441\u0438\u0440\u043e\u0432\u043a\u0430)",
            )

        # --- Group 3: Advanced Settings ---
        with st.expander("\u0420\u0430\u0441\u0448\u0438\u0440\u0435\u043d\u043d\u044b\u0435 \u043d\u0430\u0441\u0442\u0440\u043e\u0439\u043a\u0438", expanded=False):
            _default_gap = opt.gap_threshold if opt else GAP_THRESHOLD
            gap_threshold = st.slider(
                "\u041f\u043e\u0440\u043e\u0433 \u0433\u044d\u043f\u0430", min_value=0.10, max_value=0.25,
                value=float(_default_gap), step=0.025, format="%.3f",
                help="\u0418\u0441\u043a\u043b\u044e\u0447\u0435\u043d\u0438\u0435 \u0430\u043a\u0442\u0438\u0432\u043e\u0432 \u0441 \u043e\u0434\u043d\u043e\u0434\u043d\u0435\u0432\u043d\u044b\u043c \u0433\u044d\u043f\u043e\u043c > \u043f\u043e\u0440\u043e\u0433\u0430 (\u0444\u0438\u043b\u044c\u0442\u0440 \u0441\u043f\u043b\u0438\u0442\u043e\u0432/\u043e\u0442\u0447\u0451\u0442\u043e\u0432)",
            )

            _default_atr = opt.atr_period if opt else ATR_PERIOD
            atr_period = st.slider(
                "\u041f\u0435\u0440\u0438\u043e\u0434 ATR (\u0434\u043d\u0438)", min_value=10, max_value=30, step=5,
                value=_default_atr,
                help="\u041e\u043a\u043d\u043e ATR \u0434\u043b\u044f \u0440\u0430\u0441\u0447\u0451\u0442\u0430 \u0440\u0430\u0437\u043c\u0435\u0440\u0430 \u043f\u043e\u0437\u0438\u0446\u0438\u0439 (ATR risk parity)",
            )

            _default_rf = opt.risk_factor if opt else RISK_FACTOR
            risk_factor = st.slider(
                "Risk Factor", min_value=0.0005, max_value=0.002,
                value=float(_default_rf), step=0.0005, format="%.4f",
                help="\u0420\u0438\u0441\u043a \u043d\u0430 \u043f\u043e\u0437\u0438\u0446\u0438\u044e \u0432 \u0434\u0435\u043d\u044c (Clenow default: 0.001 = 10 bps)",
            )

    return {
        "data_years": data_years,
        "refresh": refresh,
        "optimize_mode": optimize_mode,
        "opt_n_trials": opt_n_trials,
        "opt_oos_days": opt_oos_days,
        "opt_min_is_days": opt_min_is_days,
        "initial_capital": float(initial_capital),
        "r2_lookback": r2_lookback,
        "top_n": top_n,
        "kama_asset_period": kama_asset_period,
        "kama_spy_period": kama_spy_period,
        "kama_buffer": kama_buffer,
        "rebal_period_weeks": rebal_period_weeks,
        "gap_threshold": gap_threshold,
        "atr_period": atr_period,
        "risk_factor": risk_factor,
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

    if kind == "wfo":
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
            is_cagrs = [s.is_metrics.get("cagr", 0) for s in wfo.steps]
            oos_cagrs = [s.oos_metrics.get("cagr", 0) for s in wfo.steps]
            avg_is = np.mean(is_cagrs)
            avg_oos = np.mean(oos_cagrs)
            if avg_is > 0:
                degradation = 1.0 - avg_oos / avg_is
                verdict = "Acceptable" if degradation <= 0.5 else "High \u2014 possible overfitting"
                st.metric("IS/OOS Degradation", f"{degradation:.1%}", verdict)
            else:
                st.metric("IS/OOS Degradation", "N/A")

            # Stitched OOS equity chart
            st.subheader("Stitched Out-of-Sample Equity")
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
                f"R\u00b2 LB={fp.r2_lookback}, KAMA Asset={fp.kama_asset_period}, "
                f"KAMA SPY={fp.kama_spy_period}, "
                f"Buffer={fp.kama_buffer}, Top N={fp.top_n}, "
                f"Rebal={fp.rebal_period_weeks}w"
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

    st.title("R\u00b2 Momentum Strategy")

    st.caption(
        "Long/Cash \u00b7 145 Cross-Asset ETFs \u00b7 R\u00b2 Momentum \u00b7 ATR Risk Parity \u00b7 KAMA Trend Filter"
    )

    if sidebar["run_clicked"]:
        universe = cached_fetch_etf()
        period = f"{sidebar['data_years']}y"
        cache_suffix = f"_etf_{period}"

        close_prices, open_prices = cached_fetch_prices(
            tuple(sorted(universe)), sidebar["refresh"],
            cache_suffix=cache_suffix, period=period,
        )

        min_days = max(sidebar["r2_lookback"], sidebar["kama_asset_period"],
                       sidebar["kama_spy_period"]) + 10
        valid = filter_valid_tickers(close_prices, min_days)

        # R² backtest kwargs
        bt_kwargs = dict(
            r2_lookback=sidebar["r2_lookback"],
            top_n=sidebar["top_n"],
            kama_asset_period=sidebar["kama_asset_period"],
            kama_spy_period=sidebar["kama_spy_period"],
            kama_buffer=sidebar["kama_buffer"],
            rebal_period_weeks=sidebar["rebal_period_weeks"],
            gap_threshold=sidebar["gap_threshold"],
            atr_period=sidebar["atr_period"],
            risk_factor=sidebar["risk_factor"],
        )

        # --- Optimization dispatch ---
        best = None
        opt_mode = sidebar["optimize_mode"]

        if opt_mode == "Walk-Forward":
            from scripts.wfo_r2_momentum import run_r2_walk_forward

            n_trials = sidebar["opt_n_trials"]
            with st.spinner(
                f"Walk-forward optimization ({n_trials} trials/step, "
                f"IS={sidebar['opt_min_is_days']}d, OOS={sidebar['opt_oos_days']}d)..."
            ):
                wfo_result = run_r2_walk_forward(
                    close_prices, open_prices, valid,
                    sidebar["initial_capital"],
                    n_trials_per_step=n_trials,
                    oos_days=sidebar["opt_oos_days"],
                    min_is_days=sidebar["opt_min_is_days"],
                )
                best = wfo_result.final_params
                st.session_state["opt_detail"] = ("wfo", wfo_result)

        if best is not None:
            st.session_state["optimized_params"] = best
            st.toast(
                f"Optimal: R\u00b2 LB={best.r2_lookback}, "
                f"KAMA={best.kama_asset_period}, "
                f"SPY={best.kama_spy_period}, "
                f"Top N={best.top_n}, "
                f"Rebal={best.rebal_period_weeks}w",
                icon="\u2705",
            )
            bt_kwargs.update(
                r2_lookback=best.r2_lookback,
                kama_asset_period=best.kama_asset_period,
                kama_spy_period=best.kama_spy_period,
                kama_buffer=best.kama_buffer,
                top_n=best.top_n,
                rebal_period_weeks=best.rebal_period_weeks,
                gap_threshold=best.gap_threshold,
                atr_period=best.atr_period,
                risk_factor=best.risk_factor,
            )
        elif opt_mode != "None":
            st.warning("Optimization found no valid combinations. Using manual parameters.")

        with st.spinner(f"Running R\u00b2 Momentum simulation on {len(valid)} tickers..."):
            equity, spy_equity, holdings_history, cash_history, trade_log = run_backtest(
                close_prices, open_prices, valid,
                initial_capital=sidebar["initial_capital"],
                **bt_kwargs,
            )

        result = BacktestResult(
            equity=equity,
            spy_equity=spy_equity,
            holdings_history=holdings_history,
            cash_history=cash_history,
            trade_log=trade_log,
        )

        st.session_state["result"] = result
        st.session_state["close_prices"] = close_prices

    if "result" not in st.session_state:
        st.info("Click **Run Backtest** in the sidebar to start.")
        return

    result: BacktestResult = st.session_state["result"]
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
        page_title="R\u00b2 Momentum Strategy",
        page_icon="\U0001F4C8",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    main()
