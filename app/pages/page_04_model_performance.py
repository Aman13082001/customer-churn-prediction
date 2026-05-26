"""Model performance page for the churn prediction dashboard."""

import streamlit as st
from sklearn.metrics import roc_curve

from utils.model_utils import (
    evaluate_all_models,
    get_best_model_name,
    get_confusion_matrix,
    get_evaluation_split,
    get_feature_importances,
    load_all_models,
)
from utils.plotting import confusion_matrix_chart, feature_importance_chart, model_comparison_chart, roc_curve_chart


def show_page() -> None:
    """Render the model performance page."""
    st.title("🎯 Model Performance")
    st.markdown("Compare all saved models, inspect ROC behavior, and highlight the strongest performer.")

    _, X_test, _, y_test = get_evaluation_split()
    metrics = evaluate_all_models(X_test, y_test)
    best_model = get_best_model_name(metrics)
    model_options = metrics["model"].tolist()

    selected_model = st.selectbox("Select model for confusion matrix and feature importance", model_options, index=model_options.index(best_model))

    best_row = metrics[metrics["model"] == best_model].iloc[0]
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Accuracy", f"{best_row['accuracy']:.4f}")
    with col2:
        st.metric("Precision", f"{best_row['precision']:.4f}")
    with col3:
        st.metric("Recall", f"{best_row['recall']:.4f}")
    with col4:
        st.metric("F1", f"{best_row['f1']:.4f}")
    with col5:
        st.metric("ROC-AUC", f"{best_row['roc_auc']:.4f}")

    st.success(f"**{best_model}** is the current best model based on ROC-AUC.")

    st.divider()

    st.subheader("📊 Model Comparison Table")
    st.dataframe(metrics.round(4), use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("📈 Metric Comparison")
    st.plotly_chart(model_comparison_chart(metrics), use_container_width=True)

    st.divider()

    st.subheader("📉 ROC Curve Comparison")
    models = load_all_models()
    roc_payload = {}
    for model_name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_payload[model_name] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "roc_auc": float(metrics.loc[metrics["model"] == model_name, "roc_auc"].iloc[0]),
        }
    st.plotly_chart(roc_curve_chart(roc_payload), use_container_width=True)

    st.divider()

    st.subheader("🔍 Selected Model Diagnostics")
    cm, _, _ = get_confusion_matrix(selected_model)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(confusion_matrix_chart(cm, ["Stayed", "Churned"], selected_model), use_container_width=True)
    with col2:
        importances = get_feature_importances(selected_model)
        if importances is None:
            st.warning(f"{selected_model} does not expose tree-based feature importances.")
        else:
            st.plotly_chart(feature_importance_chart(importances, selected_model), use_container_width=True)

    st.divider()

    st.subheader("📌 Why this model was chosen")
    st.write(
        f"{best_model} leads the leaderboard because it achieved the highest ROC-AUC score and the strongest balance of precision and recall on the held-out evaluation split."
    )