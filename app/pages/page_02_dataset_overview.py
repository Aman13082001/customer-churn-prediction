"""Dataset overview page for the churn prediction dashboard."""

import streamlit as st

from utils.data_utils import (
    get_data_type_summary,
    get_descriptive_stats,
    get_missing_summary,
    get_summary_stats,
    load_data,
)
from utils.plotting import class_distribution_chart, correlation_heatmap, feature_type_breakdown_chart


def show_page() -> None:
    """Render the dataset overview page."""
    st.title("📋 Dataset Overview")
    st.markdown("Inspect the data shape, quality, distributions, and key feature relationships.")

    df = load_data()
    stats = get_summary_stats(df)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", f"{stats['rows']:,}")
    with col2:
        st.metric("Columns", stats["columns"])
    with col3:
        st.metric("Churned", f"{stats['churned']:,}")
    with col4:
        st.metric("Retained", f"{stats['retained']:,}")

    st.divider()

    st.subheader("🧾 Data Types")
    st.dataframe(get_data_type_summary(df), use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("⚠️ Missing Values")
    missing_summary = get_missing_summary(df)
    if missing_summary.empty:
        st.success("No missing values detected in the processed dataset.")
    else:
        st.dataframe(missing_summary, use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("📈 Distribution and Correlation")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(class_distribution_chart(df), use_container_width=True)
    with col2:
        st.plotly_chart(correlation_heatmap(df), use_container_width=True)

    st.divider()

    st.subheader("📊 Feature Type Breakdown")
    st.plotly_chart(feature_type_breakdown_chart(df), use_container_width=True)

    st.divider()

    st.subheader("📋 Descriptive Statistics")
    st.dataframe(get_descriptive_stats(df), use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("� Top Numeric Correlations with Churn")
    corr_series = (
        df.select_dtypes(include=["number"])
        .corr()["Churn Value"]
        .drop("Churn Value")
        .sort_values(ascending=False)
    )
    st.dataframe(
        corr_series.rename("Correlation Score").reset_index().rename(columns={"index": "Feature"}).head(6),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    st.subheader("�👀 Interactive Data Preview")
    filter_choice = st.selectbox("Filter by churn status", ["All", "Churned", "Stayed"])
    preview_df = df.copy()
    if filter_choice == "Churned":
        preview_df = preview_df[preview_df["Churn Value"] == 1]
    elif filter_choice == "Stayed":
        preview_df = preview_df[preview_df["Churn Value"] == 0]

    st.dataframe(preview_df.head(15), use_container_width=True)

    st.divider()

    st.subheader("💡 Data Quality Notes")
    st.write(
        "The cleaned dataset is ready for modeling, and all numeric fields required by the pipeline are populated after preprocessing."
    )