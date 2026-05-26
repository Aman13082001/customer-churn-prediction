"""Home page for the churn prediction dashboard."""

import streamlit as st

from utils.data_utils import get_project_summary, load_data
from utils.model_utils import evaluate_all_models, get_best_model_name, get_evaluation_split, get_model_status


def show_page() -> None:
    """Render the home page."""
    st.title("🏠 Customer Churn Prediction")
    st.markdown(
        "A polished Streamlit dashboard that turns your churn dataset into live insights, model comparisons, and retention recommendations."
    )

    df = load_data()
    X_train, X_test, y_train, y_test = get_evaluation_split()
    metrics = evaluate_all_models(X_test, y_test)
    best_model = get_best_model_name(metrics)
    summary = get_project_summary(df, best_model)
    model_status = get_model_status()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Customers", f"{summary['total_customers']:,}")
    with col2:
        st.metric("Churn Rate", f"{summary['churn_rate']:.2f}%")
    with col3:
        st.metric("Features", summary["number_of_features"])
    with col4:
        st.metric("Best Model", best_model)

    st.divider()



    st.subheader("🧠 Model Status")
    status_cols = st.columns(2)
    with status_cols[0]:
        st.metric("Models Loaded", len(model_status))
        st.write("• Logistic Regression")
        st.write("• Decision Tree")
        st.write("• Random Forest")
        st.write("• XGBoost")
    with status_cols[1]:
        st.success(f"Best model selected: **{best_model}**")
        st.write("The dashboard uses cached model artifacts and real dataset-derived metrics.")

    st.divider()

    st.subheader("📌 What you can explore")
    col1, col2 = st.columns(2)
    with col1:
        st.write("- Dataset quality, types, and missing values")
        st.write("- Churn drivers across contract, tenure, and payment behavior")
        st.write("- Model metrics, ROC performance, and feature importance")
    with col2:
        st.write("- Risk scoring for individual customer profiles")
        st.write("- Retention recommendations based on model output")
        st.write("- End-to-end portfolio-ready visuals for stakeholder review")

    st.divider()

    st.subheader("📊 Fast Facts")
    fact_cols = st.columns(3)
    with fact_cols[0]:
        st.metric("Churned Customers", f"{summary['total_customers'] * summary['churn_rate'] / 100:.0f}")
    with fact_cols[1]:
        st.metric("Retained Customers", f"{summary['total_customers'] - summary['total_customers'] * summary['churn_rate'] / 100:.0f}")
    with fact_cols[2]:
        st.metric("Avg Monthly Charge", f"${df['Monthly Charges'].mean():.2f}")