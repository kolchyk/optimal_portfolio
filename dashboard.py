"""Streamlit dashboard for KAMA momentum strategy.

Run with: streamlit run dashboard.py
"""

import logging

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import structlog

from src.portfolio_sim.config import INITIAL_CAPITAL
from src.portfolio_sim.data import fetch_price_data, fetch_sp500_tickers
from src.portfolio_sim.engine import run_simulation
from src.portfolio_sim.reporting import compute_drawdown_series, compute_metrics

if __name__ == "__main__":
    st.set_page_config(
        page_title="KAMA Momentum Strategy",
        layout="wide",
    )


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


@st.cache_data(show_spinner="Fetching S&P 500 tickers...")
def cached_fetch_sp500():
    return fetch_sp500_tickers()


@st.cache_data(show_spinner="Downloading price data...")
def cached_fetch_prices(tickers_tuple: tuple, refresh: bool):
    return fetch_price_data(list(tickers_tuple), refresh=refresh)


def plot_equity_curve(
    equity: pd.Series,
    spy_equity: pd.Series,
    title: str = "KAMA Momentum vs S&P 500",
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity.index, y=equity.values,
        mode="lines", name="KAMA Momentum",
        line=dict(color="#2962FF", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=spy_equity.index, y=spy_equity.values,
        mode="lines", name="S&P 500 (Buy & Hold)",
        line=dict(color="#888888", width=1.5, dash="dash"),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        template="plotly_white",
        height=450,
    )
    return fig


def plot_drawdown(equity: pd.Series) -> go.Figure:
    dd = compute_drawdown_series(equity)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values * 100,
        mode="lines", name="Drawdown",
        line=dict(color="#e74c3c", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(231, 76, 60, 0.2)",
    ))
    fig.update_layout(
        title="Strategy Drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        template="plotly_white",
        height=300,
    )
    return fig


def render_metric_cards(strat_metrics: dict, spy_metrics: dict):
    st.subheader("Strategy")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CAGR", f"{strat_metrics['cagr']:.1%}")
    c2.metric("Max Drawdown", f"{-strat_metrics['max_drawdown']:.1%}")
    c3.metric("Sharpe", f"{strat_metrics['sharpe']:.2f}")
    c4.metric("Calmar", f"{strat_metrics['calmar']:.2f}")

    st.subheader("S&P 500 Benchmark")
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("CAGR", f"{spy_metrics['cagr']:.1%}")
    c6.metric("Max Drawdown", f"{-spy_metrics['max_drawdown']:.1%}")
    c7.metric("Sharpe", f"{spy_metrics['sharpe']:.2f}")
    c8.metric("Calmar", f"{spy_metrics['calmar']:.2f}")


def main():
    st.title("KAMA Momentum Strategy")
    st.caption("Long/Cash only | Equal weight | S&P 500 universe | Daily KAMA review")

    with st.sidebar:
        st.title("Settings")
        refresh = st.checkbox("Refresh data cache", value=False)
        run_clicked = st.button("Run Backtest", type="primary", use_container_width=True)

    if run_clicked:
        sp500 = cached_fetch_sp500()

        close_prices, open_prices = cached_fetch_prices(
            tuple(sorted(sp500)), refresh
        )

        min_days = 756
        valid = [
            t for t in close_prices.columns
            if t != "SPY" and len(close_prices[t].dropna()) >= min_days
        ]

        with st.spinner(f"Running simulation on {len(valid)} tickers..."):
            equity, spy_equity = run_simulation(
                close_prices, open_prices, valid, INITIAL_CAPITAL
            )

        st.session_state["equity"] = equity
        st.session_state["spy_equity"] = spy_equity

    if "equity" in st.session_state:
        equity = st.session_state["equity"]
        spy_equity = st.session_state["spy_equity"]
        strat_m = compute_metrics(equity)
        spy_m = compute_metrics(spy_equity)

        render_metric_cards(strat_m, spy_m)
        st.plotly_chart(plot_equity_curve(equity, spy_equity), use_container_width=True)
        st.plotly_chart(plot_drawdown(equity), use_container_width=True)
    else:
        st.info("Click **Run Backtest** in the sidebar to start.")


if __name__ == "__main__":
    main()
