"""Customer Risk Simulator — Prediction Page."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from app.utils.data_utils import (
    load_data,
    get_valid_categories,
    validate_prediction_input,
    preprocess_for_prediction,
    NUMERIC_COLUMNS,
    CATEGORICAL_COLUMNS,
)
from app.utils.model_utils import (
    load_all_models,
    predict_single,
    get_retention_recommendations,
)
from app.utils.ui_components import apply_global_styles, page_header, section_header

DISPLAY_NAMES = {
    "Logistic Regression": "Logistic Regression",
    "Decision Tree": "Decision Tree",
    "Random Forest": "Random Forest",
    "XGBoost": "XGBoost",
}


def show_page() -> None:
    """Render the customer prediction page."""
    apply_global_styles()
    
    page_header(
        "🔮",
        "Customer Risk Simulator",
        "Enter customer profile to predict churn probability and get retention recommendations"
    )
    
    df = load_data()
    try:
        models = load_all_models()
    except Exception as e:
        st.error(f"Could not load models: {e}")
        return
    
    model_names = list(models.keys())
    valid_categories = get_valid_categories(df)
    
    col_form, col_result = st.columns([1, 1], gap="large")
    
    with col_form:
        st.markdown("### 📋 Customer Profile")
        
        # Model selector
        model_key = st.selectbox(
            "Select prediction model:",
            model_names,
            format_func=lambda x: DISPLAY_NAMES.get(x, x)
        )
        st.markdown("---")
        
        # Account features
        with st.expander("📋 Account Information", expanded=True):
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            contract = st.selectbox(
                "Contract Type",
                valid_categories.get("Contract", ["Month-to-month"]),
                index=0
            )
            paperless = st.selectbox(
                "Paperless Billing",
                valid_categories.get("Paperless Billing", ["Yes", "No"]),
                index=0
            )
            payment = st.selectbox(
                "Payment Method",
                valid_categories.get("Payment Method", []),
                index=0 if valid_categories.get("Payment Method") else None
            )
        
        # Billing
        with st.expander("💰 Billing", expanded=True):
            monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
            max_total = round(monthly_charges * max(tenure, 1), 2)
            total_charges = st.slider(
                "Total Charges ($)",
                float(monthly_charges),
                float(max_total) if max_total > monthly_charges else float(monthly_charges) + 1000,
                float(max_total * 0.8) if max_total > monthly_charges else float(monthly_charges * 12),
                step=1.0
            )
        
        # Demographics & Services
        with st.expander("👤 Demographics", expanded=False):
            gender = st.selectbox("Gender", valid_categories.get("Gender", ["Male", "Female"]), index=0)
            senior = st.selectbox("Senior Citizen", valid_categories.get("Senior Citizen", ["No", "Yes"]), index=0)
            partner = st.selectbox("Partner", valid_categories.get("Partner", ["Yes", "No"]), index=0)
            dependents = st.selectbox("Dependents", valid_categories.get("Dependents", ["Yes", "No"]), index=0)
        
        with st.expander("📱 Internet & Services", expanded=False):
            phone = st.selectbox("Phone Service", valid_categories.get("Phone Service", ["Yes", "No"]), index=0)
            multi_lines = st.selectbox("Multiple Lines", valid_categories.get("Multiple Lines", ["No"]), index=0)
            internet = st.selectbox("Internet Service", valid_categories.get("Internet Service", ["DSL"]), index=0)
            online_sec = st.selectbox("Online Security", valid_categories.get("Online Security", ["No"]), index=0)
            online_bk = st.selectbox("Online Backup", valid_categories.get("Online Backup", ["No"]), index=0)
            device = st.selectbox("Device Protection", valid_categories.get("Device Protection", ["No"]), index=0)
            tech = st.selectbox("Tech Support", valid_categories.get("Tech Support", ["No"]), index=0)
            tv = st.selectbox("Streaming TV", valid_categories.get("Streaming TV", ["No"]), index=0)
            movies = st.selectbox("Streaming Movies", valid_categories.get("Streaming Movies", ["No"]), index=0)
        
        predict_btn = st.button("🔮 Predict Churn Risk", type="primary")
    
    with col_result:
        st.markdown("### 🎯 Prediction Result")
        
        if predict_btn:
            # Build input dictionary
            input_dict = {
                "Tenure Months": tenure,
                "Monthly Charges": monthly_charges,
                "Total Charges": total_charges,
                "Gender": gender,
                "Senior Citizen": senior,
                "Partner": partner,
                "Dependents": dependents,
                "Phone Service": phone,
                "Multiple Lines": multi_lines,
                "Internet Service": internet,
                "Online Security": online_sec,
                "Online Backup": online_bk,
                "Device Protection": device,
                "Tech Support": tech,
                "Streaming TV": tv,
                "Streaming Movies": movies,
                "Contract": contract,
                "Paperless Billing": paperless,
                "Payment Method": payment,
            }
            
            # Validate input
            is_valid, errors = validate_prediction_input(input_dict, df)
            if not is_valid:
                for err in errors:
                    st.error(f"❌ {err}")
                st.stop()
            
            # Preprocess and predict
            try:
                input_df = preprocess_for_prediction(input_dict)
                result = predict_single(model_key, input_df)
                
                prob = result["probability"]
                risk_label = result["risk_label"]
                
                # Gauge chart
                gauge_color = "#FF4B4B" if risk_label == "High Risk" else "#FFA500" if risk_label == "Medium Risk" else "#00CC76"
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=prob * 100,
                    number={"suffix": "%", "font": {"size": 40}},
                    title={"text": "Churn Probability", "font": {"size": 16}},
                    delta={"reference": 26.5, "suffix": "% vs avg"},
                    gauge={
                        "axis": {"range": [0, 100], "ticksuffix": "%"},
                        "bar": {"color": gauge_color},
                        "steps": [
                            {"range": [0, 40], "color": "#0D1117"},
                            {"range": [40, 70], "color": "#1A1A2E"},
                            {"range": [70, 100], "color": "#2D0A0A"},
                        ],
                        "threshold": {"line": {"color": "white", "width": 2}, "value": 26.5}
                    }
                ))
                fig_gauge.update_layout(
                    template="plotly_dark",
                    height=280,
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(t=50, b=0, l=20, r=20)
                )
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Risk badge
                badge_class = {
                    "High Risk": "badge-high",
                    "Medium Risk": "badge-medium",
                    "Low Risk": "badge-low"
                }[risk_label]
                st.markdown(
                    f'<div class="{badge_class}">{risk_label}</div>',
                    unsafe_allow_html=True
                )
                
                st.markdown("---")
                st.markdown("#### 💡 Retention Recommendations")
                
                recs = get_retention_recommendations(input_df, prob)
                for i, rec in enumerate(recs, 1):
                    st.markdown(f"**{i}. {rec}**")
                
                # Model info footer
                st.caption(f"Prediction by: {DISPLAY_NAMES.get(model_key, model_key)} | Confidence: {result['confidence']:.1%}")
                
            except Exception as e:
                st.error(f"Prediction error: {e}")
        else:
            st.markdown("""
            <div style='text-align:center;padding:2rem'>
                <div style='font-size:48px;margin-bottom:1rem'>🔮</div>
                <div style='color:#9AA4B2'>Fill in the customer profile and click</div>
                <div style='color:#E6EDF3;font-weight:600;font-size:1.1rem'>Predict Churn Risk</div>
                <div style='color:#9AA4B2;margin-top:0.5rem'>to see results</div>
            </div>
            """, unsafe_allow_html=True)
