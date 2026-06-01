"""Main Streamlit dashboard application."""

import streamlit as st
import sys
from pathlib import Path

# Fix path issues
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.utils.data_utils import load_data
from app.utils.model_utils import load_all_models
from app.utils.ui_components import apply_global_styles
from app.pages import (
    page_01_home,
    page_02_dataset_overview,
    page_03_churn_analysis,
    page_04_model_performance,
    page_05_prediction,
)

PAGES = {
    "🏠 Home": page_01_home,
    "📁 Dataset Overview": page_02_dataset_overview,
    "📈 Churn Analysis": page_03_churn_analysis,
    "🤖 Model Performance": page_04_model_performance,
    "🔮 Customer Prediction": page_05_prediction,
}

def render_sidebar():
    with st.sidebar:
        # --- ADDED CSS TO HIDE NATIVE NAV ---
        st.markdown("""
            <style>
                [data-testid="stSidebarNav"] {
                    display: none !important;
                }
            </style>
        """, unsafe_allow_html=True)
        # ------------------------------------

        st.markdown("""
        ## 📊 ChurnIQ
        Customer Intelligence Platform
        """, unsafe_allow_html=True)

        # Your custom radio navigation
        page = st.radio("Navigation", list(PAGES.keys()), label_visibility="visible") # Changed to visible so users see the 'Navigation' label

        st.markdown("---")
        try:
            df = load_data()
            models = load_all_models()
            st.markdown(f"""
            👥 **{len(df):,}** customers
            ⚠️ **{df['Churn'].mean()*100:.1f}%** churn rate
            🤖 **{len(models)}** models active
            """, unsafe_allow_html=True)
        except Exception:
            st.caption("Dataset Source: IBM Telecom Customer Churn Dataset obtained from Kaggle, containing customer demographics, account information, service usage patterns, and churn status for predictive analytics.")

        st.markdown("---")
        st.markdown("This dashboard analyzes customer behavior and predicts the likelihood of churn, helping organizations improve customer retention strategies.")

    return page


def main():
    st.set_page_config(
        page_title="ChurnIQ — Customer Intelligence Platform",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Optional: Apply global styles if you have a helper function
    apply_global_styles()

    if "initialized" not in st.session_state:
        st.session_state.initialized = True

    try:
        page = render_sidebar()
        PAGES[page].show_page()
    except Exception as e:
        st.error(f"⚠️ Page error: {str(e)}")


if __name__ == "__main__":
    main()