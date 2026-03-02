"""Streamlit dashboard for Hybrid R² Momentum + Vol-Targeting strategy.

Run with: streamlit run dashboard.py
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import structlog

from src.portfolio_sim.config import (
    DEFAULT_N_TRIALS,
    INITIAL_CAPITAL,
)
from src.portfolio_sim.cli_utils import filter_valid_tickers
from src.portfolio_sim.data import fetch_etf_tickers, fetch_price_data
from src.portfolio_sim.engine import run_simulation
from src.portfolio_sim.models import SimulationResult
from src.portfolio_sim.params import StrategyParams
from src.portfolio_sim.indicators import compute_kama_series
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
CHART_TEMPLATE = "plotly_dark"
COLOR_STRATEGY = "#00B4D8"
COLOR_BENCHMARK = "#64748B"
COLOR_POSITIVE = "#10B981"
COLOR_NEGATIVE = "#EF4444"
COLOR_DRAWDOWN = "#F43F5E"
COLOR_PORTFOLIO = "#22D3EE"


def _inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        div[data-testid="stHorizontalBlock"] {
            align-items: stretch;
        }
        div[data-testid="stColumn"] > div:first-child,
        div[data-testid="stColumn"] [data-testid="stVerticalBlockBorderWrapper"],
        div[data-testid="stColumn"] [data-testid="stVerticalBlockBorderWrapper"] > div,
        div[data-testid="stColumn"] [data-testid="stVerticalBlock"],
        div[data-testid="stColumn"] [data-testid="stElementContainer"] {
            height: 100%;
        }
        .metric-card {
            background: #111827;
            border: 1px solid #1F2937;
            border-radius: 6px;
            padding: 16px 20px;
            margin: 4px 0;
            height: 100%;
            box-sizing: border-box;
            transition: border-color 0.2s ease;
        }
        .metric-card:hover {
            border-color: #374151;
        }
        .metric-card .label {
            color: #94A3B8;
            font-size: 0.72rem;
            margin-bottom: 4px;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 500;
        }
        .metric-card .value {
            font-size: 1.4rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', 'SF Mono', monospace;
            letter-spacing: -0.5px;
        }
        .metric-card.positive { border-left: 3px solid #10B981; }
        .metric-card.positive .value { color: #10B981; }
        .metric-card.negative { border-left: 3px solid #EF4444; }
        .metric-card.negative .value { color: #EF4444; }
        .metric-card.neutral { border-left: 3px solid #64748B; }
        .metric-card.neutral .value { color: #E2E8F0; }
        .section-divider {
            border-top: 1px solid #1F2937;
            margin: 2rem 0 1.5rem 0;
        }
        .section-header {
            color: #94A3B8;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            font-weight: 600;
            margin-bottom: 0.75rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid #1F2937;
        }
        [data-testid="stSidebar"] {
            border-right: 1px solid #1F2937;
        }
        [data-testid="stSidebar"] .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #00B4D8 0%, #0284C7 100%);
            border: none;
            font-weight: 600;
            letter-spacing: 0.5px;
            text-transform: uppercase;
            font-size: 0.85rem;
            padding: 0.6rem 1rem;
        }
        [data-testid="stSidebar"] .stButton > button[kind="secondary"] {
            border: 1px solid #374151;
            color: #94A3B8;
            background: transparent;
            font-weight: 500;
            letter-spacing: 0.5px;
            text-transform: uppercase;
            font-size: 0.75rem;
            padding: 0.5rem 1rem;
        }
        [data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
            border-color: #00B4D8;
            color: #E2E8F0;
        }
        .stDataFrame {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
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
        logging.getLogger(
            "streamlit.runtime.scriptrunner_utils.script_run_context"
        ).setLevel(logging.ERROR)
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
# Parameter fingerprint for auto-rerun
# ---------------------------------------------------------------------------
def _param_fingerprint(sidebar: dict) -> tuple:
    """Return a hashable fingerprint of all strategy + data parameters."""
    return (
        sidebar["data_years"],
        sidebar["initial_capital"],
        sidebar["top_n"],
        sidebar["kama_asset_period"],
        sidebar["kama_buffer"],
        sidebar["atr_period"],
        sidebar["risk_factor"],
        sidebar["max_per_class"],
        sidebar["target_vol"],
        sidebar["max_leverage"],
        sidebar["portfolio_vol_lookback"],
        sidebar["min_invested_pct"],
    )


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
# Charts
# ---------------------------------------------------------------------------
def _base_layout(**kwargs) -> dict:
    defaults = dict(
        template=CHART_TEMPLATE,
        margin=dict(l=50, r=20, t=50, b=40),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(size=11, color="#94A3B8"),
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0D1117",
        font=dict(family="Inter, sans-serif", color="#94A3B8", size=12),
        title=dict(font=dict(size=14, color="#E2E8F0"), x=0, xanchor="left"),
        xaxis=dict(gridcolor="#1E293B", zerolinecolor="#1E293B", linecolor="#1F2937"),
        yaxis=dict(gridcolor="#1E293B", zerolinecolor="#1E293B", linecolor="#1F2937"),
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
            name="Strategy",
            line=dict(color=COLOR_STRATEGY, width=2.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=result.spy_equity.index,
            y=result.spy_equity.values,
            mode="lines",
            name="S&P 500 (Buy & Hold)",
            line=dict(color=COLOR_BENCHMARK, width=1.5, dash="dot"),
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
            xaxis=dict(type="date"),
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
            fillcolor="rgba(244, 63, 94, 0.15)",
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
            fillcolor="rgba(100, 116, 139, 0.08)",
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
            textfont=dict(size=11, color="#E2E8F0"),
            colorscale=[
                [0, "#991B1B"], [0.25, "#DC2626"],
                [0.45, "#1E293B"], [0.55, "#1E293B"],
                [0.75, "#059669"], [1, "#047857"],
            ],
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
            marker_line_width=0,
        )
    )
    fig.add_trace(
        go.Bar(
            x=[str(y) for y in years],
            y=[spy_yr.get(y, 0) for y in years],
            name="S&P 500",
            marker_color=COLOR_BENCHMARK,
            marker_line_width=0,
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
    fig.add_hline(y=0, line_dash="dot", line_color="#374151", line_width=1)
    fig.add_hline(y=1, line_dash="dash", line_color="#1E293B", line_width=1,
                  annotation_text="Good", annotation_font_color="#64748B",
                  annotation_font_size=10)
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
            marker=dict(line=dict(color="#0A0E17", width=2)),
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


def plot_asset_signals(
    ticker: str,
    close_prices: pd.DataFrame,
    kama_period: int,
    kama_buffer: float,
    trade_log: list[dict],
) -> go.Figure:
    """Price chart with KAMA, buffer bands, and buy/sell markers for one asset."""
    prices = close_prices[ticker].dropna()
    kama = compute_kama_series(prices, period=kama_period)
    upper_band = kama * (1 + kama_buffer)
    lower_band = kama * (1 - kama_buffer)

    fig = go.Figure()

    # Close price
    fig.add_trace(
        go.Scatter(
            x=prices.index, y=prices.values,
            mode="lines", name="Close",
            line=dict(color="#E2E8F0", width=1.5),
        )
    )

    # KAMA line
    fig.add_trace(
        go.Scatter(
            x=kama.index, y=kama.values,
            mode="lines", name=f"KAMA({kama_period})",
            line=dict(color="#F59E0B", width=2),
        )
    )

    # Upper band (entry threshold)
    fig.add_trace(
        go.Scatter(
            x=upper_band.index, y=upper_band.values,
            mode="lines", name="Entry threshold",
            line=dict(color=COLOR_POSITIVE, width=1, dash="dash"),
            opacity=0.4,
        )
    )

    # Lower band (exit threshold)
    fig.add_trace(
        go.Scatter(
            x=lower_band.index, y=lower_band.values,
            mode="lines", name="Exit threshold",
            line=dict(color=COLOR_NEGATIVE, width=1, dash="dash"),
            opacity=0.4,
        )
    )

    # Trade markers
    trades_df = pd.DataFrame(trade_log)
    if not trades_df.empty:
        trades_df["date"] = pd.to_datetime(trades_df["date"])
        ticker_trades = trades_df[trades_df["ticker"] == ticker]

        buys = ticker_trades[ticker_trades["action"] == "buy"]
        sells = ticker_trades[ticker_trades["action"].isin(["sell", "trim"])]

        if not buys.empty:
            buy_dates = buys["date"]
            buy_prices = prices.reindex(buy_dates).values
            fig.add_trace(
                go.Scatter(
                    x=buy_dates, y=buy_prices,
                    mode="markers", name="Buy",
                    marker=dict(
                        symbol="triangle-up", size=12,
                        color=COLOR_POSITIVE,
                        line=dict(width=1, color="#0A0E17"),
                    ),
                    hovertemplate=(
                        "<b>BUY %{x|%Y-%m-%d}</b><br>"
                        "Close: $%{y:.2f}<br>"
                        "Exec: $" + buys["price"].apply(lambda p: f"{p:.2f}").values
                        + "<br>Shares: " + buys["shares"].apply(lambda s: f"{s:.1f}").values
                        + "<extra></extra>"
                    ),
                )
            )

        if not sells.empty:
            sell_dates = sells["date"]
            sell_prices = prices.reindex(sell_dates).values
            fig.add_trace(
                go.Scatter(
                    x=sell_dates, y=sell_prices,
                    mode="markers", name="Sell",
                    marker=dict(
                        symbol="triangle-down", size=12,
                        color=COLOR_NEGATIVE,
                        line=dict(width=1, color="#0A0E17"),
                    ),
                    hovertemplate=(
                        "<b>SELL %{x|%Y-%m-%d}</b><br>"
                        "Close: $%{y:.2f}<br>"
                        "Exec: $" + sells["price"].apply(lambda p: f"{p:.2f}").values
                        + "<br>Shares: " + sells["shares"].apply(lambda s: f"{s:.1f}").values
                        + "<extra></extra>"
                    ),
                )
            )

    fig.update_layout(
        **_base_layout(
            title=f"{ticker} — Price, KAMA & Trade Signals",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=480,
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
_DEFAULTS = StrategyParams()


def _render_sidebar() -> dict:
    with st.sidebar:
        st.markdown(
            '<h2 style="margin:0; padding:0; color:#E2E8F0; font-weight:700;">'
            'Optimal Portfolio</h2>'
            '<p style="color:#64748B; font-size:0.8rem; margin-top:4px;">'
            'R² Momentum Backtester</p>',
            unsafe_allow_html=True,
        )

        optimize_clicked = st.button(
            "Run Optimization",
            type="primary",
            width="stretch",
            help="Walk-Forward Optimization: find best parameters on IS window, validate on OOS.",
        )

        # --- Group 1: Data Settings ---
        with st.expander("Data Settings", expanded=True):
            data_years = st.slider(
                "Data Period (years)",
                min_value=3, max_value=5, value=3,
                help="Number of years of historical data to load",
            )

            refresh = st.checkbox("Refresh data cache", value=False)

        # --- Group 2: Optimization Settings ---
        with st.expander("Optimization Settings", expanded=False):
            opt_n_trials = st.slider(
                "Optuna Trials (per step)", min_value=20, max_value=500,
                value=DEFAULT_N_TRIALS, step=10,
            )
            opt_oos_days = st.slider(
                "OOS Window (days)", min_value=10, max_value=63,
                value=63, step=7,
                help="Out-of-sample validation window for each WFO step.",
            )
            opt_min_is_days = st.slider(
                "IS Window (days)", min_value=63, max_value=252,
                value=126, step=21,
                help="Minimum in-sample optimization window.",
            )

        # --- Group 3: Strategy Parameters ---
        with st.expander("Strategy Parameters", expanded=True):
            initial_capital = st.number_input(
                "Initial Capital ($)",
                min_value=1_000.0, max_value=10_000_000.0,
                value=float(INITIAL_CAPITAL), step=1_000.0,
                help="Starting portfolio value",
            )

            opt = st.session_state.get("optimized_params")

            _default_r2 = opt.r2_window if opt else _DEFAULTS.r2_window
            st.caption(f"R² Window: {_default_r2}d")

            _default_top_n = opt.top_n if opt else _DEFAULTS.top_n
            top_n = st.slider(
                "Top N Assets",
                min_value=5, max_value=25, step=5,
                value=min(_default_top_n, 25),
                help="Maximum number of positions held simultaneously",
            )

            _default_kama = opt.kama_asset_period if opt else _DEFAULTS.kama_asset_period
            kama_asset_period = st.slider(
                "KAMA Period (asset)", min_value=10, max_value=50, value=_default_kama,
                help="KAMA for individual asset trend filter (trading days)",
            )

            _default_buffer = opt.kama_buffer if opt else _DEFAULTS.kama_buffer
            kama_buffer = st.slider(
                "KAMA Buffer", min_value=0.005, max_value=0.05,
                value=float(_default_buffer), step=0.005, format="%.3f",
                help="Hysteresis threshold to prevent false regime switches",
            )

            st.caption(f"Rebalancing: every {_DEFAULTS.rebal_days} trading days")

        # --- Group 4: Advanced / Vol-Targeting ---
        with st.expander("Advanced Settings", expanded=False):
            _default_atr = opt.atr_period if opt else _DEFAULTS.atr_period
            atr_period = st.slider(
                "ATR Period (days)", min_value=10, max_value=30, step=5,
                value=_default_atr,
                help="ATR window for position sizing",
            )

            _default_rf = opt.risk_factor if opt else _DEFAULTS.risk_factor
            risk_factor = st.slider(
                "Risk Factor", min_value=0.0005, max_value=0.002,
                value=float(_default_rf), step=0.0005, format="%.4f",
                help="Daily risk per position (Clenow default: 0.001 = 10 bps)",
            )

            _default_mpc = opt.max_per_class if opt else _DEFAULTS.max_per_class
            max_per_class = st.slider(
                "Max Per Asset Class", min_value=1, max_value=10, step=1,
                value=_default_mpc,
                help="Maximum positions from the same asset class",
            )

            _default_tvol = opt.target_vol if opt else _DEFAULTS.target_vol
            target_vol = st.slider(
                "Target Vol (annual)", min_value=0.04, max_value=0.50,
                value=float(_default_tvol), step=0.02, format="%.2f",
                help="Target annual portfolio volatility",
            )

            _default_mlev = opt.max_leverage if opt else _DEFAULTS.max_leverage
            max_leverage = st.slider(
                "Max Leverage", min_value=1.0, max_value=2.0,
                value=float(_default_mlev), step=0.1, format="%.1f",
                help="Maximum scaling factor for vol-targeting overlay",
            )

            _default_pvlb = opt.portfolio_vol_lookback if opt else _DEFAULTS.portfolio_vol_lookback
            portfolio_vol_lookback = st.slider(
                "Vol Lookback (days)", min_value=15, max_value=35, step=5,
                value=_default_pvlb,
                help="Window for estimating realized portfolio volatility",
            )

            _default_mip = opt.min_invested_pct if opt else _DEFAULTS.min_invested_pct
            min_invested_pct_int = st.slider(
                "Min Invested %", min_value=0, max_value=90,
                value=int(round(float(_default_mip) * 100)), step=10,
                format="%d%%",
                help="Floor for invested fraction. 0 = disabled (vol-targeting fully controls). "
                     "80 = at least 80% of equity always invested.",
            )
            min_invested_pct = min_invested_pct_int / 100.0

    return {
        "data_years": data_years,
        "refresh": refresh,
        "opt_n_trials": opt_n_trials,
        "opt_oos_days": opt_oos_days,
        "opt_min_is_days": opt_min_is_days,
        "initial_capital": float(initial_capital),
        "top_n": top_n,
        "kama_asset_period": kama_asset_period,
        "kama_buffer": kama_buffer,
        "atr_period": atr_period,
        "risk_factor": risk_factor,
        "max_per_class": max_per_class,
        "target_vol": target_vol,
        "max_leverage": max_leverage,
        "portfolio_vol_lookback": portfolio_vol_lookback,
        "min_invested_pct": min_invested_pct,
        "optimize_clicked": optimize_clicked,
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

            st.subheader("Stitched Out-of-Sample Equity")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=wfo.stitched_equity.index,
                y=wfo.stitched_equity.values,
                name="Strategy (OOS)",
                line=dict(width=2, color=COLOR_STRATEGY),
            ))
            if wfo.stitched_spy_equity is not None and not wfo.stitched_spy_equity.empty:
                fig.add_trace(go.Scatter(
                    x=wfo.stitched_spy_equity.index,
                    y=wfo.stitched_spy_equity.values,
                    name="SPY (OOS)",
                    line=dict(width=1, dash="dot", color=COLOR_BENCHMARK),
                ))
            fig.update_layout(
                **_base_layout(
                    yaxis_title="Portfolio Value ($)",
                    height=400,
                )
            )
            st.plotly_chart(fig, width="stretch")

            fp = wfo.final_params
            st.info(
                f"Recommended live params: "
                f"KAMA Asset={fp.kama_asset_period}, "
                f"Buffer={fp.kama_buffer}, Top N={fp.top_n}, "
                f"Target Vol={fp.target_vol:.0%}, "
                f"Max Lev={fp.max_leverage}, "
                f"Max/Class={fp.max_per_class}"
            )


# ---------------------------------------------------------------------------
# Data helper
# ---------------------------------------------------------------------------
def _fetch_data(sidebar: dict, cached_fetch_etf, cached_fetch_prices):
    """Fetch universe and price data based on sidebar settings."""
    universe = cached_fetch_etf()
    period = f"{sidebar['data_years']}y"
    cache_suffix = f"_etf_{period}"
    close_prices, open_prices, high_prices, low_prices = cached_fetch_prices(
        tuple(sorted(universe)), sidebar["refresh"],
        cache_suffix=cache_suffix, period=period,
    )
    return close_prices, open_prices, high_prices, low_prices


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    _setup_logging()

    cached_fetch_etf = st.cache_data(show_spinner="Fetching ETF tickers...")(_fetch_etf_impl)
    cached_fetch_prices = st.cache_data(show_spinner="Downloading price data...")(_fetch_prices_impl)

    _inject_css()
    sidebar = _render_sidebar()

    st.title("Backtest Results")

    st.caption(
        "Long/Cash \u00b7 Cross-Asset ETFs \u00b7 R\u00b2 Momentum \u00b7 ATR Risk Parity \u00b7 Vol-Targeting"
    )

    # --- Path 1: Run Optimization (slow, button-gated) ---
    if sidebar["optimize_clicked"]:
        close_prices, open_prices, high_prices, low_prices = _fetch_data(sidebar, cached_fetch_etf, cached_fetch_prices)

        params = StrategyParams()
        valid = filter_valid_tickers(close_prices, params.warmup)

        from src.portfolio_sim.walk_forward import run_walk_forward

        n_trials = sidebar["opt_n_trials"]
        try:
            with st.spinner(
                f"Walk-forward optimization ({n_trials} trials/step, "
                f"IS={sidebar['opt_min_is_days']}d, OOS={sidebar['opt_oos_days']}d)..."
            ):
                wfo_result = run_walk_forward(
                    close_prices, open_prices, valid,
                    sidebar["initial_capital"],
                    n_trials_per_step=n_trials,
                    oos_days=sidebar["opt_oos_days"],
                    min_is_days=sidebar["opt_min_is_days"],
                    high_prices=high_prices,
                    low_prices=low_prices,
                )
                best = wfo_result.final_params
                st.session_state["opt_detail"] = ("wfo", wfo_result)

            st.session_state["optimized_params"] = best
            st.toast(
                f"Optimal: KAMA={best.kama_asset_period}, "
                f"Top N={best.top_n}, "
                f"Vol={best.target_vol:.0%}, "
                f"Max/Class={best.max_per_class}",
                icon="\u2705",
            )

            with st.spinner(f"Running simulation with optimized params on {len(valid)} tickers..."):
                result = run_simulation(
                    close_prices, open_prices, valid,
                    initial_capital=sidebar["initial_capital"],
                    params=best,
                    high_prices=high_prices,
                    low_prices=low_prices,
                )

            st.session_state["result"] = result
            st.session_state["close_prices"] = close_prices
            st.session_state["params"] = best
            st.session_state["_param_hash"] = _param_fingerprint(sidebar)

        except ValueError as e:
            st.error(f"Optimization failed: {e}")

    # --- Path 2: Auto-rerun backtest when parameters change ---
    else:
        current_hash = _param_fingerprint(sidebar)
        prev_hash = st.session_state.get("_param_hash")

        if sidebar["refresh"]:
            prev_hash = None  # force rerun on data refresh

        if current_hash != prev_hash or "result" not in st.session_state:
            close_prices, open_prices, high_prices, low_prices = _fetch_data(sidebar, cached_fetch_etf, cached_fetch_prices)

            params = StrategyParams(
                top_n=sidebar["top_n"],
                kama_asset_period=sidebar["kama_asset_period"],
                kama_buffer=sidebar["kama_buffer"],
                atr_period=sidebar["atr_period"],
                risk_factor=sidebar["risk_factor"],
                max_per_class=sidebar["max_per_class"],
                target_vol=sidebar["target_vol"],
                max_leverage=sidebar["max_leverage"],
                portfolio_vol_lookback=sidebar["portfolio_vol_lookback"],
                min_invested_pct=sidebar["min_invested_pct"],
            )

            valid = filter_valid_tickers(close_prices, params.warmup)

            with st.spinner(f"Running simulation on {len(valid)} tickers..."):
                result = run_simulation(
                    close_prices, open_prices, valid,
                    initial_capital=sidebar["initial_capital"],
                    params=params,
                    high_prices=high_prices,
                    low_prices=low_prices,
                )

            st.session_state["result"] = result
            st.session_state["close_prices"] = close_prices
            st.session_state["params"] = params
            st.session_state["_param_hash"] = current_hash

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
    st.markdown('<div class="section-header">PERFORMANCE METRICS</div>', unsafe_allow_html=True)
    _render_metric_row("Strategy", strat_m)
    _render_metric_row("S&P 500 Benchmark", spy_m)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # --- Section 2: Equity Curve ---
    st.markdown('<div class="section-header">EQUITY CURVE</div>', unsafe_allow_html=True)
    log_scale = st.checkbox("Log scale", value=False)
    st.plotly_chart(
        plot_equity_curve(result, log_scale=log_scale), width="stretch"
    )

    # --- Section 4: Drawdown ---
    st.markdown('<div class="section-header">DRAWDOWN ANALYSIS</div>', unsafe_allow_html=True)
    st.plotly_chart(plot_drawdowns(result), width="stretch")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # --- Section 5: Monthly Heatmap + Yearly Bar ---
    st.markdown('<div class="section-header">RETURN ANALYSIS</div>', unsafe_allow_html=True)
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
    st.markdown('<div class="section-header">RISK-ADJUSTED RETURNS</div>', unsafe_allow_html=True)
    st.plotly_chart(
        plot_rolling_sharpe(result.equity, result.spy_equity),
        width="stretch",
    )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # --- Section 7: Portfolio Composition ---
    st.markdown('<div class="section-header">PORTFOLIO COMPOSITION</div>', unsafe_allow_html=True)
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

    # --- Section 8: Asset Signal Analysis ---
    st.markdown('<div class="section-header">SIGNAL ANALYSIS</div>', unsafe_allow_html=True)
    if result.trade_log:
        traded_tickers = sorted({t["ticker"] for t in result.trade_log})
        params: StrategyParams = st.session_state.get("params", StrategyParams())
        selected_ticker = st.selectbox(
            "Select asset", traded_tickers, index=0,
        )
        st.plotly_chart(
            plot_asset_signals(
                selected_ticker, close_prices,
                kama_period=params.kama_asset_period,
                kama_buffer=params.kama_buffer,
                trade_log=result.trade_log,
            ),
            width="stretch",
        )
    else:
        st.info("No trades recorded — run a backtest first.")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # --- Section 9: Trade Log ---
    st.markdown('<div class="section-header">TRADE LOG</div>', unsafe_allow_html=True)
    with st.expander("Trade Log", expanded=False):
        _render_trade_log(result)


if __name__ == "__main__":
    st.set_page_config(
        page_title="Optimal Portfolio | R² Momentum",
        page_icon="\U0001F4C8",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    main()
