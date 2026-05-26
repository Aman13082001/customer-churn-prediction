"""Customer prediction page for the churn prediction dashboard."""

import streamlit as st

from utils.data_utils import load_data, preprocess_for_prediction
from utils.model_utils import get_retention_recommendations, get_risk_label, load_all_models, predict_single
from utils.plotting import probability_gauge_chart


def show_page() -> None:
    """Render the customer prediction page."""
    st.title("🔮 Customer Prediction")
    st.markdown("Simulate churn risk for a customer profile and review retention actions tailored to the scenario.")

    df = load_data()
    model_names = list(load_all_models().keys())

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"], index=0)
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"], index=0)
            partner = st.selectbox("Partner", ["No", "Yes"], index=0)
            dependents = st.selectbox("Dependents", ["No", "Yes"], index=0)
        with col2:
            phone_service = st.selectbox("Phone Service", ["No", "Yes"], index=0)
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"], index=0)
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], index=0)
            online_security = st.selectbox("Online Security", ["No", "Yes"], index=0)
        with col3:
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], index=0)
            payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], index=0)
            paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"], index=0)
            model_name = st.selectbox("Model", model_names, index=model_names.index("XGBoost"))

        tenure_months = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.slider("Monthly Charges", 0.0, 150.0, 70.0, 0.5)
        total_charges = st.slider("Total Charges", 0.0, 10000.0, 500.0, 10.0)

        submitted = st.form_submit_button("🔍 Predict Churn", use_container_width=True)

    if submitted:
        input_dict = {
            "Gender": gender,
            "Senior Citizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "Tenure Months": float(tenure_months),
            "Phone Service": phone_service,
            "Multiple Lines": multiple_lines,
            "Internet Service": internet_service,
            "Online Security": online_security,
            "Online Backup": "No",
            "Device Protection": "No",
            "Tech Support": "No",
            "Streaming TV": "No",
            "Streaming Movies": "No",
            "Contract": contract,
            "Paperless Billing": paperless_billing,
            "Payment Method": payment_method,
            "Monthly Charges": float(monthly_charges),
            "Total Charges": float(total_charges),
        }

        result = predict_single(model_name, input_dict)
        input_frame = preprocess_for_prediction(input_dict)
        recommendations = get_retention_recommendations(input_frame, result["probability"])

        st.divider()
        st.subheader("🎯 Probability Outcome")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(probability_gauge_chart(result["probability"]), use_container_width=True)
        with col2:
            st.metric("Risk Label", result["risk_label"])
            st.metric("Confidence", f"{result['confidence']:.2%}")
            st.metric("Selected Model", result["model"])

        if result["risk_label"] == "High Risk":
            st.error("High churn risk detected. Immediate retention action is recommended.")
        elif result["risk_label"] == "Medium Risk":
            st.warning("Moderate churn risk detected. Engagement and pricing review are recommended.")
        else:
            st.success("Low churn risk detected. Continue standard retention support.")

        st.divider()

        st.subheader("💡 Top Retention Recommendations")
        for idx, recommendation in enumerate(recommendations, start=1):
            st.write(f"{idx}. {recommendation}")

        st.divider()

        st.subheader("📊 Customer Profile Summary")
        profile_cols = st.columns(3)
        with profile_cols[0]:
            st.write(f"**Gender:** {gender}")
            st.write(f"**Senior Citizen:** {senior_citizen}")
            st.write(f"**Partner:** {partner}")
        with profile_cols[1]:
            st.write(f"**Dependents:** {dependents}")
            st.write(f"**Tenure:** {tenure_months} months")
            st.write(f"**Phone Service:** {phone_service}")
        with profile_cols[2]:
            st.write(f"**Internet Service:** {internet_service}")
            st.write(f"**Contract:** {contract}")
            st.write(f"**Monthly Charges:** ${monthly_charges:.2f}")

        st.divider()

        st.subheader("📌 Business Interpretation")
        st.write(
            f"Using {result['model']}, this customer profile is estimated to have a {result['probability']:.1%} probability of churn."
        )