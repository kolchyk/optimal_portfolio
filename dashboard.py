"""Streamlit dashboard for portfolio simulation.

Run with: streamlit run dashboard.py
"""

import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import structlog

from src.portfolio_sim.config import INITIAL_CAPITAL, StrategyParams
from src.portfolio_sim.data import fetch_price_data, load_tickers
from src.portfolio_sim.engine import run_simulation
from src.portfolio_sim.reporting import compute_drawdown_series, compute_metrics
from src.portfolio_sim.walk_forward import run_walk_forward

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Portfolio Simulator",
    page_icon="📊",
    layout="wide",
)


def _setup_logging():
    """Configure structlog once."""
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
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading ticker universe...")
def cached_load_tickers():
    return load_tickers()


@st.cache_data(show_spinner="Downloading price data...")
def cached_fetch_prices(tickers_tuple: tuple, refresh: bool):
    return fetch_price_data(list(tickers_tuple), refresh=refresh)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar():
    st.sidebar.title("Configuration")

    mode = st.sidebar.radio(
        "Mode",
        ["Single Run", "Walk-Forward Validation"],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Strategy Parameters")
    kama_period = st.sidebar.select_slider(
        "KAMA Period", options=list(range(5, 22, 2)), value=10
    )
    lookback_period = st.sidebar.select_slider(
        "Lookback Period", options=list(range(10, 43, 5)), value=21
    )
    top_n_selection = st.sidebar.select_slider(
        "Top N Selection", options=list(range(5, 21, 5)), value=10
    )

    params = StrategyParams(
        kama_period=kama_period,
        lookback_period=lookback_period,
        top_n_selection=top_n_selection,
    )

    st.sidebar.markdown("---")
    n_trials = st.sidebar.number_input(
        "Optuna Trials (WFV)", min_value=5, max_value=500, value=50, step=5
    )
    metric = st.sidebar.selectbox(
        "Optimization Metric", ["calmar", "sharpe", "return"], index=0
    )

    refresh_data = st.sidebar.checkbox("Refresh data cache", value=False)

    run_clicked = st.sidebar.button("Run Simulation", type="primary", use_container_width=True)

    return {
        "mode": mode,
        "params": params,
        "n_trials": n_trials,
        "metric": metric,
        "refresh": refresh_data,
        "run": run_clicked,
    }


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def plot_equity_curve(equity: pd.Series, title: str = "Equity Curve") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=equity.index,
            y=equity.values,
            mode="lines",
            name="Portfolio",
            line=dict(color="#e74c3c", width=2),
            fill="tozeroy",
            fillcolor="rgba(231, 76, 60, 0.1)",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        template="plotly_white",
        height=400,
        margin=dict(l=50, r=20, t=40, b=40),
    )
    return fig


def plot_drawdown(equity: pd.Series) -> go.Figure:
    dd = compute_drawdown_series(equity)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dd.index,
            y=dd.values * 100,
            mode="lines",
            name="Drawdown",
            line=dict(color="#e67e22", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(230, 126, 34, 0.2)",
        )
    )
    fig.update_layout(
        title="Drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        template="plotly_white",
        height=300,
        margin=dict(l=50, r=20, t=40, b=40),
    )
    return fig


def plot_wfv_window_returns(windows: list[dict]) -> go.Figure:
    names = [f"W{w['window']}" for w in windows]
    returns = [w["oos_return_pct"] for w in windows]
    colors = ["#2ecc71" if r >= 0 else "#e74c3c" for r in returns]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=names, y=returns, marker_color=colors, name="OOS Return")
    )
    fig.update_layout(
        title="OOS Return by Window",
        xaxis_title="Window",
        yaxis_title="Return (%)",
        template="plotly_white",
        height=350,
        margin=dict(l=50, r=20, t=40, b=40),
    )
    return fig


def plot_holdings_bar(weights: np.ndarray, tickers: list[str]) -> go.Figure:
    weight_s = pd.Series(weights, index=tickers)
    active = weight_s[weight_s.abs() > 0.001].sort_values()

    if active.empty:
        fig = go.Figure()
        fig.add_annotation(text="No active holdings", showarrow=False)
        return fig

    colors = ["#2ecc71" if w > 0 else "#e74c3c" for w in active.values]
    fig = go.Figure(
        go.Bar(
            x=active.values * 100,
            y=active.index,
            orientation="h",
            marker_color=colors,
        )
    )
    fig.update_layout(
        title="Portfolio Allocation (Long/Short)",
        xaxis_title="Weight (%)",
        template="plotly_white",
        height=max(300, len(active) * 25 + 100),
        margin=dict(l=80, r=20, t=40, b=40),
    )
    return fig


def plot_net_exposure(net_exp: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=net_exp.index,
            y=net_exp.values * 100,
            mode="lines",
            name="Net Exposure",
            line=dict(color="#3498db", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(52, 152, 219, 0.15)",
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title="Net Exposure (Market Breathing)",
        xaxis_title="Date",
        yaxis_title="Net Exposure (%)",
        template="plotly_white",
        height=250,
        margin=dict(l=50, r=20, t=40, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Metric cards
# ---------------------------------------------------------------------------
def render_metric_cards(metrics: dict, avg_net_exp: float | None = None):
    if avg_net_exp is not None:
        c1, c2, c3, c4, c5 = st.columns(5)
        c5.metric("Avg Net Exp", f"{avg_net_exp:.1%}")
    else:
        c1, c2, c3, c4 = st.columns(4)
    c1.metric("CAGR", f"{metrics['cagr']:.1%}")
    c2.metric("Max Drawdown", f"{-metrics['max_drawdown']:.1%}")
    c3.metric("Sharpe", f"{metrics['sharpe']:.2f}")
    c4.metric("Calmar", f"{metrics['calmar']:.2f}")


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main():
    st.title("Portfolio Simulator Dashboard")

    sidebar_cfg = render_sidebar()

    # Load data eagerly
    tickers_600, original_portfolio = cached_load_tickers()
    full_list = tuple(sorted(set(tickers_600 + list(original_portfolio.keys()))))

    if sidebar_cfg["run"]:
        close_prices, open_prices = cached_fetch_prices(
            full_list, sidebar_cfg["refresh"]
        )

        # Filter tickers with sufficient history
        min_len = 756
        valid = [
            t
            for t in close_prices.columns
            if len(close_prices[t].dropna()) >= min_len
        ]
        close_prices = close_prices[valid].ffill().bfill()
        open_prices = open_prices[valid].ffill().bfill()
        all_tickers = [t for t in valid if t != "SPY"]

        if sidebar_cfg["mode"] == "Walk-Forward Validation":
            with st.spinner("Running Walk-Forward Validation... This may take a while."):
                wfv_result = run_walk_forward(
                    close_prices,
                    open_prices,
                    all_tickers,
                    n_trials=sidebar_cfg["n_trials"],
                    metric=sidebar_cfg["metric"],
                )
            st.session_state["wfv_result"] = wfv_result
            st.session_state["mode"] = "wfv"
        else:
            with st.spinner("Running simulation..."):
                params = sidebar_cfg["params"]
                lookback_buffer = params.lookback_period + 5
                sim_len = min(756, len(close_prices) - lookback_buffer)
                sim_start = close_prices.index[-sim_len]

                sim_prices = close_prices.loc[sim_start:]
                sim_open = open_prices.loc[sim_start:]

                equity, gross_exp, net_exp, weights = run_simulation(
                    sim_prices,
                    sim_open,
                    close_prices,
                    all_tickers,
                    params,
                    INITIAL_CAPITAL,
                )

            eq_series = pd.Series(
                equity, index=sim_prices.index[: len(equity)]
            )
            net_exp_series = pd.Series(
                net_exp, index=sim_prices.index[: len(net_exp)]
            )
            st.session_state["equity"] = eq_series
            st.session_state["net_exposures"] = net_exp_series
            st.session_state["weights"] = weights
            st.session_state["tickers"] = all_tickers
            st.session_state["mode"] = "single"

    # ---- Render results ----
    if "mode" not in st.session_state:
        st.info("Configure parameters in the sidebar and click **Run Simulation** to start.")
        return

    if st.session_state["mode"] == "single":
        _render_single_run()
    elif st.session_state["mode"] == "wfv":
        _render_wfv()


def _render_single_run():
    equity = st.session_state["equity"]
    weights = st.session_state["weights"]
    tickers = st.session_state["tickers"]
    net_exp = st.session_state.get("net_exposures", pd.Series(dtype=float))
    metrics = compute_metrics(equity)

    avg_net = float(net_exp.mean()) if not net_exp.empty else None
    tab_perf, tab_holdings = st.tabs(["Performance", "Portfolio"])

    with tab_perf:
        render_metric_cards(metrics, avg_net_exp=avg_net)
        st.plotly_chart(plot_equity_curve(equity), use_container_width=True)
        st.plotly_chart(plot_drawdown(equity), use_container_width=True)
        if not net_exp.empty:
            st.plotly_chart(plot_net_exposure(net_exp), use_container_width=True)

    with tab_holdings:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                plot_holdings_bar(weights, tickers), use_container_width=True
            )
        with col2:
            weight_s = pd.Series(weights, index=tickers)
            active = weight_s[weight_s.abs() > 0.001].sort_values(
                key=lambda x: x.abs(), ascending=False
            )
            if not active.empty:
                df = pd.DataFrame(
                    {"Ticker": active.index, "Weight": active.values}
                )
                df["Side"] = df["Weight"].map(lambda w: "LONG" if w > 0 else "SHORT")
                df["Weight"] = df["Weight"].map("{:.1%}".format)
                st.dataframe(df, hide_index=True, use_container_width=True)
            else:
                st.write("No active holdings.")


def _render_wfv():
    wfv = st.session_state["wfv_result"]
    oos_equity = wfv["oos_equity"]
    windows = wfv["windows"]
    oos_metrics = compute_metrics(oos_equity)

    tab_perf, tab_wfv, tab_holdings = st.tabs(
        ["Performance", "WFV Analysis", "Portfolio"]
    )

    with tab_perf:
        render_metric_cards(oos_metrics)
        st.plotly_chart(
            plot_equity_curve(oos_equity, "OOS Equity Curve (Walk-Forward)"),
            use_container_width=True,
        )
        st.plotly_chart(plot_drawdown(oos_equity), use_container_width=True)

        # Additional stats
        c1, c2 = st.columns(2)
        c1.metric("Total Return", f"{oos_metrics['total_return']:.1%}")
        c2.metric("Ann. Volatility", f"{oos_metrics['annualized_vol']:.1%}")

    with tab_wfv:
        st.plotly_chart(
            plot_wfv_window_returns(windows), use_container_width=True
        )

        # Window breakdown table
        rows = []
        for w in windows:
            rows.append(
                {
                    "Window": w["window"],
                    "Train": f"{w['train_start']} -> {w['train_end']}",
                    "Test": f"{w['test_start']} -> {w['test_end']}",
                    "IS Score": f"{w['is_score']:.4f}",
                    "OOS Return": f"{w['oos_return_pct']:.1f}%",
                    "OOS MaxDD": f"{w['oos_max_dd_pct']:.1f}%",
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        # Parameters per window (expandable)
        for w in windows:
            with st.expander(f"Window {w['window']} Parameters"):
                st.json(w["params"])

        # Win rate
        wins = sum(1 for w in windows if w["oos_return_pct"] > 0)
        st.metric("WFV Win Rate", f"{wins}/{len(windows)} ({wins / len(windows):.0%})")

    with tab_holdings:
        st.info(
            "In WFV mode, portfolio composition varies by window. "
            "See the WFV Analysis tab for per-window details."
        )


if __name__ == "__main__":
    main()
