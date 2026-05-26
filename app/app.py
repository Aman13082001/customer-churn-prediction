"""Main Streamlit dashboard application."""

import streamlit as st

from pages.page_01_home import show_page as page_01_home
from pages.page_02_dataset_overview import show_page as page_02_dataset_overview
from pages.page_03_churn_analysis import show_page as page_03_churn_analysis
from pages.page_04_model_performance import show_page as page_04_model_performance
from pages.page_05_prediction import show_page as page_05_prediction


PAGE_OPTIONS = {
    "Home": page_01_home,
    "Dataset Overview": page_02_dataset_overview,
    "Churn Analysis": page_03_churn_analysis,
    "Model Performance": page_04_model_performance,
    "Customer Prediction": page_05_prediction,
}


st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CSS TO HIDE DEFAULT NAVIGATION ---
st.markdown(
    """
    <style>
    [data-testid="stSidebarNav"] {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def main() -> None:
    """Render the selected dashboard page."""
    current_page = st.session_state.get("current_page", "Home")
    if current_page not in PAGE_OPTIONS:
        current_page = "Home"

    st.sidebar.title("Navigation")
    selected_page = st.sidebar.radio(
        "Select a page",
        list(PAGE_OPTIONS.keys()),
        index=list(PAGE_OPTIONS.keys()).index(current_page),
    )
    st.session_state.current_page = selected_page

    PAGE_OPTIONS[selected_page]()


if __name__ == "__main__":
    main()
