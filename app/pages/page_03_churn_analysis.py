"""Churn Analysis — Churn Intelligence Center."""

import streamlit as st
import pandas as pd
import plotly.express as px
from app.utils.data_utils import load_data, get_churn_analysis_data, TARGET_COLUMN
from app.utils.ui_components import apply_global_styles, page_header, metric_card_row, insight_card, section_header


def show_page() -> None:
    """Render the churn analysis page."""
    apply_global_styles()
    
    df = load_data()
    analysis = get_churn_analysis_data(df)
    churn_rate = df[TARGET_COLUMN].mean() * 100
    
    page_header(
        "📈",
        "Churn Analysis",
        "Identify patterns and segments at highest risk"
    )
    
    # Key metrics
    metric_card_row([
        {"label": "Overall Churn Rate", "value": f"{churn_rate:.1f}%"},
        {"label": "Churned Customers", "value": f"{int(df[TARGET_COLUMN].sum()):,}"},
        {"label": "Retained Customers", "value": f"{int(len(df) - df[TARGET_COLUMN].sum()):,}"},
        {"label": "Retention Rate", "value": f"{100-churn_rate:.1f}%"},
    ])
    
    st.markdown("---")
    
    # Tabs for different analyses
    tabs = st.tabs(["🥧 Overview", "📋 By Contract", "⏳ By Tenure", "💰 By Charges", "🌐 By Services"])
    
    # TAB 1: Overview
    with tabs[0]:
        section_header("Churn Distribution Overview")
        col1, col2 = st.columns(2)
        
        with col1:
            # Donut chart
            churn_counts = df[TARGET_COLUMN].value_counts().sort_index()
            fig_donut = px.pie(
                values=churn_counts.values,
                names=["Retained", "Churned"],
                hole=0.4,
                title="Churn Status Distribution",
                template="plotly_dark"
            )
            fig_donut.update_layout(paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_donut, width='stretch')
        
        with col2:
            st.metric("Total Retained", f"{int(churn_counts.get(0, 0)):,}")
            st.metric("Total Churned", f"{int(churn_counts.get(1, 0)):,}")
            st.metric("Churn Rate", f"{churn_rate:.1f}%")
    
    # TAB 2: By Contract
    with tabs[1]:
        section_header("Churn Rate by Contract Type")
        if not analysis["contract_rate"].empty:
            fig_contract = px.bar(
                analysis["contract_rate"],
                x="Contract",
                y="churn_rate",
                title="Churn Rate by Contract Type",
                labels={"churn_rate": "Churn Rate", "Contract": "Contract Type"},
                template="plotly_dark",
                color="churn_rate",
                color_continuous_scale="Reds"
            )
            fig_contract.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_contract, width='stretch')
            
            insight_card("**Month-to-month contracts show the highest churn rate.** Contract type is a key retention lever — encouraging customers to upgrade from month-to-month to annual contracts can significantly reduce churn.")
    
    # TAB 3: By Tenure
    with tabs[2]:
        section_header("Churn Rate by Customer Tenure")
        if not analysis["tenure_rate"].empty:
            fig_tenure = px.line(
                analysis["tenure_rate"],
                x="tenure_group",
                y="churn_rate",
                title="Churn Rate by Tenure Group",
                labels={"churn_rate": "Churn Rate", "tenure_group": "Tenure Group"},
                template="plotly_dark",
                markers=True
            )
            fig_tenure.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_tenure, width='stretch')
            
            insight_card("**Early-stage customers (0-12 months) churn at much higher rates.** Focus retention efforts on the first year to improve lifetime value.")
    
    # TAB 4: By Charges
    with tabs[3]:
        section_header("Churn Patterns by Billing")
        col_charges1, col_charges2 = st.columns(2)
        
        with col_charges1:
            fig_monthly = px.histogram(
                df,
                x="Monthly Charges",
                color=TARGET_COLUMN,
                nbins=30,
                title="Monthly Charges Distribution by Churn",
                template="plotly_dark"
            )
            fig_monthly.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_monthly, width='stretch')
        
        with col_charges2:
            fig_total = px.box(
                df,
                x=TARGET_COLUMN,
                y="Total Charges",
                title="Total Charges by Churn Status",
                template="plotly_dark"
            )
            fig_total.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_total, width='stretch')
        
        insight_card("**Pricing sensitivity is a factor.** Customers paying higher monthly charges may have competing alternatives or higher expectations for service quality.")
    
    # TAB 5: By Services
    with tabs[4]:
        section_header("Churn Patterns by Service Type")
        col_svc1, col_svc2 = st.columns(2)
        
        with col_svc1:
            if "Internet Service" in df.columns:
                internet_churn = df.groupby("Internet Service", observed=True)[TARGET_COLUMN].mean().reset_index()
                fig_internet = px.bar(
                    internet_churn,
                    x="Internet Service",
                    y=TARGET_COLUMN,
                    title="Churn Rate by Internet Service",
                    template="plotly_dark",
                    color=TARGET_COLUMN,
                    color_continuous_scale="Reds"
                )
                fig_internet.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_internet, width='stretch')
        
        with col_svc2:
            if "Payment Method" in df.columns:
                payment_churn = df.groupby("Payment Method", observed=True)[TARGET_COLUMN].mean().reset_index()
                fig_payment = px.bar(
                    payment_churn,
                    x="Payment Method",
                    y=TARGET_COLUMN,
                    title="Churn Rate by Payment Method",
                    template="plotly_dark",
                    color=TARGET_COLUMN,
                    color_continuous_scale="Reds"
                )
                fig_payment.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_payment, width='stretch')
        
        insight_card("**Service type and payment method affect retention.** Fiber optic customers may require special attention to competitive positioning.")

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
        st.dataframe(analysis["contract_rate"].sort_values("churn_rate", ascending=False), width="stretch", hide_index=True)
    with segment_cols[1]:
        st.write("**Payment Method Segments**")
        st.dataframe(analysis["payment_rate"].sort_values("churn_rate", ascending=False), width="stretch", hide_index=True)
    with segment_cols[2]:
        st.write("**Internet Service Segments**")
        st.dataframe(analysis["internet_rate"].sort_values("churn_rate", ascending=False), width="stretch", hide_index=True)

    st.divider()

    st.subheader("🧠 Strategic Recommendations")
    st.write(
        "- Prioritize month-to-month customers with retention incentives and early engagement.\n"
        "- Target high-charge customers with pricing and plan-review outreach.\n"
        "- Build automated campaigns for customers in the shortest tenure buckets."
    )
