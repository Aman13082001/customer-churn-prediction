"""Dataset Overview — Data Intelligence Center."""

import streamlit as st
import pandas as pd
from app.utils.data_utils import (
    load_data,
    get_summary_stats,
    get_data_type_summary,
    get_descriptive_stats,
    get_missing_summary,
    NUMERIC_COLUMNS,
    CATEGORICAL_COLUMNS,
    TARGET_COLUMN,
)
from app.utils.ui_components import apply_global_styles, page_header, metric_card_row, insight_card, section_header


def show_page() -> None:
    """Render the dataset overview page."""
    apply_global_styles()
    
    df = load_data()
    stats = get_summary_stats(df)
    
    page_header(
        "📁",
        "Dataset Overview",
        "Explore data quality, distributions, and feature relationships"
    )
    
    # Key metrics
    metric_card_row([
        {"label": "Total Rows", "value": f"{stats['rows']:,}"},
        {"label": "Total Columns", "value": str(stats['columns'])},
        {"label": "Churned", "value": f"{stats['churned']:,}"},
        {"label": "Retained", "value": f"{stats['retained']:,}"},
    ])
    
    st.markdown("---")
    
    # Tabs for different views
    tabs = st.tabs(["📋 Schema", "📊 Distributions", "🔥 Correlations", "🔍 Data Explorer"])
    
    # TAB 1: Schema
    with tabs[0]:
        section_header("Data Schema & Types")
        schema_df = get_data_type_summary(df)
        st.dataframe(schema_df, use_container_width=True, hide_index=True)
        
        missing_df = get_missing_summary(df)
        if missing_df.empty:
            st.success("✅ No missing values detected in the processed dataset.")
        else:
            st.warning("⚠️ Missing Values Detected")
            st.dataframe(missing_df, use_container_width=True, hide_index=True)
    
    # TAB 2: Distributions
    with tabs[1]:
        section_header("Feature Distributions")
        
        # Numeric distributions
        st.subheader("Numeric Features")
        col_dist1, col_dist2 = st.columns(2)
        with col_dist1:
            for col in NUMERIC_COLUMNS[:2]:
                if col in df.columns:
                    st.write(f"**{col}**")
                    st.bar_chart(df[col].value_counts().sort_index().head(20))
        with col_dist2:
            for col in NUMERIC_COLUMNS[2:]:
                if col in df.columns:
                    st.write(f"**{col}**")
                    st.bar_chart(df[col].value_counts().sort_index().head(20))
        
        # Categorical distributions
        st.subheader("Categorical Features (Sample)")
        for col in CATEGORICAL_COLUMNS[:6]:
            if col in df.columns:
                st.write(f"**{col}**")
                st.bar_chart(df[col].value_counts())
    
    # TAB 3: Correlations
    with tabs[2]:
        section_header("Feature Correlations with Churn")
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            
            numeric_df = df[NUMERIC_COLUMNS + [TARGET_COLUMN]].copy()
            correlations = numeric_df.corr()[TARGET_COLUMN].drop(TARGET_COLUMN).sort_values(ascending=False)
            
            fig = px.bar(
                x=correlations.values,
                y=correlations.index,
                orientation='h',
                title="Correlation with Churn",
                labels={'x': 'Correlation', 'y': 'Feature'},
                template="plotly_dark",
                color=correlations.values,
                color_continuous_scale="RdBu"
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
            
            st.write("**Top Correlated Features:**")
            st.dataframe(
                pd.DataFrame({
                    "Feature": correlations.index,
                    "Correlation": correlations.values.round(4)
                }),
                use_container_width=True,
                hide_index=True
            )
        except Exception as e:
            st.error(f"Could not compute correlations: {e}")
    
    # TAB 4: Data Explorer
    with tabs[3]:
        section_header("Data Explorer")
        st.write(f"Showing all {len(df):,} rows")
        st.dataframe(df, use_container_width=True)
        
        # Download option
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name="churn_data.csv",
            mime="text/csv"
        )

    st.divider()

    st.divider()

    st.subheader(" Descriptive Statistics")
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
