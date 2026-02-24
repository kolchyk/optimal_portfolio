"""Streamlit dashboard for KAMA momentum strategy.

Run with: streamlit run dashboard.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import structlog

from src.portfolio_sim.config import (
    INITIAL_CAPITAL,
    KAMA_BUFFER,
    KAMA_PERIOD,
    LOOKBACK_PERIOD,
    TOP_N,
)
from src.portfolio_sim.data import fetch_price_data, fetch_sp500_tickers
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

st.set_page_config(
    page_title="KAMA Momentum Strategy",
    page_icon="\U0001F4C8",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Theme / CSS
# ---------------------------------------------------------------------------
CHART_TEMPLATE = "plotly_dark"
COLOR_STRATEGY = "#2962FF"
COLOR_BENCHMARK = "#888888"
COLOR_POSITIVE = "#00c853"
COLOR_NEGATIVE = "#ff1744"
COLOR_DRAWDOWN = "#e74c3c"
COLOR_BULL_BG = "rgba(0, 200, 83, 0.07)"
COLOR_BEAR_BG = "rgba(255, 23, 68, 0.07)"


def _inject_css():
    st.markdown(
        """
        <style>
        .metric-card {
            background: #1e1e2f;
            border-radius: 8px;
            padding: 14px 18px;
            margin: 4px 0;
        }
        .metric-card .label {
            color: #999;
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
        .metric-card.neutral .value { color: #e0e0e0; }
        .section-divider {
            border-top: 1px solid #333;
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


_setup_logging()


# ---------------------------------------------------------------------------
# Cached data helpers
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Fetching S&P 500 tickers...")
def cached_fetch_sp500():
    return fetch_sp500_tickers()


@st.cache_data(show_spinner="Downloading price data...")
def cached_fetch_prices(tickers_tuple: tuple, refresh: bool):
    return fetch_price_data(list(tickers_tuple), refresh=refresh)


@st.cache_data
def load_sector_map() -> pd.DataFrame:
    csv_path = Path("sp500_companies.csv")
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
        return df[["Symbol", "Sector"]].dropna()
    return pd.DataFrame(columns=["Symbol", "Sector"])


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

    # Regime shading
    regime = result.regime_history
    changes = regime.ne(regime.shift()).cumsum()
    for _, group in regime.groupby(changes):
        start = group.index[0]
        end = group.index[-1]
        color = COLOR_BULL_BG if group.iloc[0] else COLOR_BEAR_BG
        fig.add_vrect(x0=start, x1=end, fillcolor=color, line_width=0, layer="below")

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
            marker=dict(line=dict(color="#1e1e2f", width=1)),
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

        st.markdown("---")
        st.subheader("Data")
        refresh = st.checkbox("Refresh data cache", value=False)

        st.markdown("---")
        st.subheader("Strategy Parameters")
        initial_capital = st.number_input(
            "Initial Capital ($)", min_value=1_000.0, max_value=10_000_000.0,
            value=float(INITIAL_CAPITAL), step=1_000.0,
        )
        top_n = st.slider("Top N Stocks", min_value=3, max_value=50, value=TOP_N)
        kama_period = st.slider("KAMA Period", min_value=5, max_value=50, value=KAMA_PERIOD)
        lookback_period = st.slider(
            "Lookback Period", min_value=20, max_value=252, value=LOOKBACK_PERIOD
        )
        kama_buffer = st.slider(
            "KAMA Buffer", min_value=0.0, max_value=0.05,
            value=float(KAMA_BUFFER), step=0.001, format="%.3f",
        )
        use_risk_adjusted = st.checkbox(
            "Risk-Adjusted Momentum",
            value=False,
            help="Rank by return/volatility instead of raw return. "
                 "Prefers smooth uptrends but may miss volatile winners.",
        )

        st.markdown("---")
        st.subheader("Universe Filter")
        sector_map = load_sector_map()
        all_sectors = sorted(sector_map["Sector"].unique())
        selected_sectors = st.multiselect(
            "Sectors", all_sectors, default=all_sectors,
        )

        # Filter tickers by selected sectors
        if selected_sectors:
            available_tickers = sorted(
                sector_map[sector_map["Sector"].isin(selected_sectors)]["Symbol"].tolist()
            )
        else:
            available_tickers = sorted(sector_map["Symbol"].tolist())

        selected_tickers = st.multiselect(
            "Tickers", available_tickers, default=available_tickers,
            help="Select specific tickers or leave all selected.",
        )

        st.markdown("---")
        run_clicked = st.button(
            "Run Backtest", type="primary", width="stretch"
        )

    return {
        "refresh": refresh,
        "initial_capital": float(initial_capital),
        "top_n": top_n,
        "kama_period": kama_period,
        "lookback_period": lookback_period,
        "kama_buffer": kama_buffer,
        "use_risk_adjusted": use_risk_adjusted,
        "selected_tickers": selected_tickers,
        "run_clicked": run_clicked,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    _inject_css()
    sidebar = _render_sidebar()

    st.title("KAMA Momentum Strategy")
    st.caption(
        "Long/Cash only \u2022 Equal weight \u2022 S&P 500 universe \u2022 Daily KAMA review"
    )

    if sidebar["run_clicked"]:
        sp500 = cached_fetch_sp500()

        # Apply ticker filter
        if sidebar["selected_tickers"]:
            universe = [t for t in sp500 if t in sidebar["selected_tickers"]]
        else:
            universe = sp500

        close_prices, open_prices = cached_fetch_prices(
            tuple(sorted(universe)), sidebar["refresh"]
        )

        min_days = 756
        valid = [
            t
            for t in close_prices.columns
            if t != "SPY" and len(close_prices[t].dropna()) >= min_days
        ]

        params = StrategyParams(
            kama_period=sidebar["kama_period"],
            lookback_period=sidebar["lookback_period"],
            top_n=sidebar["top_n"],
            kama_buffer=sidebar["kama_buffer"],
            use_risk_adjusted=sidebar["use_risk_adjusted"],
        )

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
    main()
