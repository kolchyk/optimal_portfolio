"""Multi-page Streamlit app entrypoint.

Run with: uv run streamlit run app.py
Or: uv run python app.py (redirects to streamlit run)
"""

import sys

# Redirect plain "python app.py" to "streamlit run app.py" to avoid bare-mode warnings
if __name__ == "__main__":
    from streamlit import runtime
    from streamlit.web import cli as stcli

    if not runtime.exists():
        sys.argv = ["streamlit", "run", __file__, *sys.argv[1:]]
        sys.exit(stcli.main())

import streamlit as st

st.set_page_config(
    page_title="SK Portfolio v 1.0",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

from pages.strategy_rules import page as strategy_rules_page  # noqa: E402


def backtest_page():
    """Wrapper that imports and runs the original dashboard logic."""
    from dashboard import main as dashboard_main

    dashboard_main()


pg = st.navigation(
    [
        st.Page(
            backtest_page,
            title="Backtest",
            icon=":material/monitoring:",
            default=True,
        ),
        st.Page(
            strategy_rules_page,
            title="How It Works",
            icon=":material/school:",
        ),
    ]
)
pg.run()
