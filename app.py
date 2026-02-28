"""Multi-page Streamlit app entrypoint.

Run with: streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="R\u00b2 Momentum Strategy",
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
            title="\u041f\u0440\u0430\u0432\u0438\u043b\u0430 \u0441\u0442\u0440\u0430\u0442\u0435\u0433\u0456\u0457",
            icon=":material/menu_book:",
        ),
    ]
)
pg.run()
