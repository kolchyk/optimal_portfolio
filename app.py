"""Multi-page Streamlit app entrypoint.

Run with: streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="KAMA Momentum Strategy",
    layout="wide",
)

from pages.strategy_rules import page as strategy_rules_page  # noqa: E402


def backtest_page():
    """Wrapper that imports and runs the original dashboard logic."""
    from dashboard import main as dashboard_main

    dashboard_main()


pg = st.navigation(
    [
        st.Page(backtest_page, title="Backtest", icon=":material/monitoring:"),
        st.Page(
            strategy_rules_page,
            title="Правила стратегії",
            icon=":material/menu_book:",
        ),
    ]
)
pg.run()
