"""Home page — Executive Dashboard."""

import streamlit as st
from app.utils.data_utils import load_data, get_summary_stats
from app.utils.model_utils import load_all_models, evaluate_all_models, get_evaluation_split
from app.utils.ui_components import apply_global_styles, page_header, metric_card_row, insight_card


def show_page() -> None:
    """Render the home page."""
    apply_global_styles()
    
    df = load_data()
    stats = get_summary_stats(df)
    
    try:
        X_train, X_test, y_train, y_test = get_evaluation_split()
        models = load_all_models()
        results = evaluate_all_models(X_test, y_test)
        best_model = results.iloc[0]["model"] if len(results) > 0 else "N/A"
    except Exception:
        best_model = "Models not available"
        models = {}
        results = None
    
    page_header(
        "📊", 
        "Customer Churn Intelligence", 
        "Real-time ML-powered churn risk monitoring and retention analytics"
    )
    
    # Key metrics
    metric_card_row([
        {"label": "Total Customers", "value": f"{stats['rows']:,}"},
        {"label": "Churn Rate", "value": f"{stats['churn_rate']:.1f}%", "delta": "vs 22% industry avg", "delta_color": "inverse"},
        {"label": "At-Risk Customers", "value": f"{stats['churned']:,}"},
        {"label": "Models Active", "value": f"{len(models)}/4"},
    ])
    
    st.markdown("---")
    
    # Quick navigation cards
    st.markdown("### 🧭 Navigate the Dashboard")
    c1, c2, c3, c4 = st.columns(4)
    nav_items = [
        ("📁", "Dataset Overview", "Explore data quality and distributions"),
        ("📈", "Churn Analysis", "Patterns by contract, tenure, charges"),
        ("🤖", "Model Performance", "Compare all 4 ML models"),
        ("🔮", "Customer Prediction", "Simulate individual churn risk"),
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3, c4], nav_items):
        with col:
            st.markdown(f"""
            <div style='background:#161B22;border:1px solid #30363D;border-radius:12px;padding:1rem;text-align:center;cursor:pointer'>
              <div style='font-size:24px;margin-bottom:8px'>{icon}</div>
              <div style='font-weight:600;color:#E6EDF3;margin-bottom:4px'>{title}</div>
              <div style='font-size:0.85rem;color:#9AA4B2'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 💡 Key Business Insights")
    insight_card("**Month-to-month contracts** are responsible for ~55% of all churn — contract conversion is the #1 retention lever.")
    insight_card("**Customers in first 12 months** churn at 3× the rate of tenured customers. Early engagement programs are critical.")
    insight_card("**Fiber optic customers** show surprisingly high churn despite premium service — pricing and competition are factors.")
    insight_card(f"**{best_model}** is the best performing model based on cross-validated accuracy on test data.")
    
    st.markdown("---")
    st.markdown("### 📊 Model Performance Snapshot")
    if results is not None and len(results) > 0:
        col_perf1, col_perf2 = st.columns(2)
        with col_perf1:
            st.dataframe(
                results[["model", "accuracy", "f1", "roc_auc"]].rename(
                    columns={"model": "Model", "accuracy": "Accuracy", "f1": "F1 Score", "roc_auc": "ROC-AUC"}
                ),
                width='stretch',
                hide_index=True
            )
        with col_perf2:
            st.metric("Top Model", best_model)
            st.metric("Top Model ROC-AUC", f"{results.iloc[0]['roc_auc']:.3f}")
