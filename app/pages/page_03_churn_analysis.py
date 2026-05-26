"""Churn analysis page for the churn prediction dashboard."""

import streamlit as st

from utils.data_utils import get_churn_analysis_data, load_data
from utils.plotting import (
    contract_churn_rate_chart,
    internet_service_churn_rate_chart,
    monthly_charge_distribution_chart,
    payment_method_churn_rate_chart,
    tenure_churn_rate_chart,
    tenure_monthly_scatter_chart,
)


def show_page() -> None:
    """Render the churn analysis page."""
    st.title("📊 Churn Analysis")
    st.markdown("Understand the strongest churn drivers and customer segments at risk.")

    df = load_data()
    analysis = get_churn_analysis_data(df)
    churn_rate = df["Churn Value"].mean() * 100

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Overall Churn Rate", f"{churn_rate:.2f}%")
    with col2:
        st.metric("Highest Risk Contract", analysis["contract_rate"]["Contract"].iloc[analysis["contract_rate"]["churn_rate"].idxmax()])
    with col3:
        st.metric("Most Common Churn Group", analysis["tenure_rate"]["tenure_group"].iloc[analysis["tenure_rate"]["churn_rate"].idxmax()])

    st.divider()

    st.subheader("📈 Churn Drivers")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(contract_churn_rate_chart(df), use_container_width=True)
    with col2:
        st.plotly_chart(tenure_churn_rate_chart(df), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(payment_method_churn_rate_chart(df), use_container_width=True)
    with col2:
        st.plotly_chart(internet_service_churn_rate_chart(df), use_container_width=True)

    st.divider()

    st.subheader("💸 Charge Behavior and Segmentation")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(monthly_charge_distribution_chart(df), use_container_width=True)
    with col2:
        st.plotly_chart(tenure_monthly_scatter_chart(df), use_container_width=True)

    st.divider()

    st.subheader("💡 Key Insights")
    contract_row = analysis["contract_rate"].sort_values("churn_rate", ascending=False).iloc[0]
    tenure_row = analysis["tenure_rate"].sort_values("churn_rate", ascending=False).iloc[0]

    insight_cols = st.columns(3)
    with insight_cols[0]:
        st.success(f"{contract_row['Contract']} customers show the highest churn rate at {contract_row['churn_rate']:.2%}.")
    with insight_cols[1]:
        st.warning(f"The highest-risk tenure segment is {tenure_row['tenure_group']} with {tenure_row['churn_rate']:.2%} churn.")
    with insight_cols[2]:
        st.info("Customers with higher monthly charges and shorter tenures are more likely to churn.")

    st.divider()

    st.subheader("📋 Churn Segment Tables")
    segment_cols = st.columns(3)
    with segment_cols[0]:
        st.write("**Contract Segments**")
        st.dataframe(analysis["contract_rate"].sort_values("churn_rate", ascending=False), use_container_width=True, hide_index=True)
    with segment_cols[1]:
        st.write("**Payment Method Segments**")
        st.dataframe(analysis["payment_rate"].sort_values("churn_rate", ascending=False), use_container_width=True, hide_index=True)
    with segment_cols[2]:
        st.write("**Internet Service Segments**")
        st.dataframe(analysis["internet_rate"].sort_values("churn_rate", ascending=False), use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("🧠 Strategic Recommendations")
    st.write(
        "- Prioritize month-to-month customers with retention incentives and early engagement.\n"
        "- Target high-charge customers with pricing and plan-review outreach.\n"
        "- Build automated campaigns for customers in the shortest tenure buckets."
    )