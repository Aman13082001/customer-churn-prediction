"""Model Performance — ML Model Comparison Center."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

from app.utils.data_utils import FEATURE_COLUMNS, TARGET_COLUMN, load_data
from app.utils.model_utils import load_all_models, evaluate_all_models, get_evaluation_split, get_feature_importances, get_confusion_matrix
from app.utils.ui_components import apply_global_styles, page_header, metric_card_row, section_header
from app.utils.plotting import confusion_matrix_chart, feature_importance_chart

DISPLAY_NAMES = {
    "Logistic Regression": "Logistic Regression",
    "Decision Tree": "Decision Tree",
    "Random Forest": "Random Forest",
    "XGBoost": "XGBoost",
}


def show_page() -> None:
    """Render the model performance page."""
    apply_global_styles()
    
    page_header(
        "🤖",
        "Model Performance Center",
        "Compare all 4 trained models across key ML evaluation metrics"
    )
    
    try:
        X_train, X_test, y_train, y_test = get_evaluation_split()
        models = load_all_models()
        results = evaluate_all_models(X_test, y_test)
    except Exception as e:
        st.error(f"Could not load models: {e}")
        return
    
    # Best model metrics
    best_model = results.iloc[0]["model"]
    best_row = results.iloc[0]
    
    metric_card_row([
        {"label": "Best Model", "value": best_model},
        {"label": "Best Accuracy", "value": f"{best_row['accuracy']:.3f}"},
        {"label": "Best ROC-AUC", "value": f"{best_row['roc_auc']:.3f}"},
        {"label": "Models Compared", "value": f"{len(results)}/4"},
    ])
    
    st.markdown("---")
    
    # Tabs for different views
    tabs = st.tabs(["📊 Comparison Table", "📈 ROC Curves", "🔢 Confusion Matrix", "🎯 Feature Importance"])
    
    # TAB 1: Comparison Table
    with tabs[0]:
        section_header("All Models — Metrics Comparison")
        
        # Display results table with highlighting
        display_df = results[["model", "accuracy", "precision", "recall", "f1", "roc_auc"]].copy()
        display_df.columns = ["Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
        
        st.dataframe(
            display_df,
            use_container_width=True
        )
        
        # Bar chart comparison
        fig = go.Figure()
        metrics_list = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        colors = ["#E63946", "#2EC4B6", "#F4A261", "#A8DADC", "#457B9D"]
        
        for i, model_name in enumerate(results["model"]):
            fig.add_trace(go.Bar(
                name=model_name,
                x=["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
                y=[results[results["model"]==model_name][m].values[0] for m in metrics_list],
                marker_color=colors[i % len(colors)]
            ))
        
        fig.update_layout(
            barmode="group",
            template="plotly_dark",
            title="Model Metrics Comparison",
            xaxis_title="Metric",
            yaxis_title="Score",
            legend=dict(orientation="h", y=-0.15),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # TAB 2: ROC Curves
    with tabs[1]:
        section_header("ROC Curves — All Models")
        
        fig_roc = go.Figure()
        fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                          line=dict(color="gray", dash="dash"))
        
        colors = ["#E63946", "#2EC4B6", "#F4A261", "#457B9D"]
        for i, (name, model) in enumerate(models.items()):
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc = roc_auc_score(y_test, y_prob)
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr,
                name=f"{name} (AUC={auc:.3f})",
                mode="lines",
                line=dict(color=colors[i], width=2)
            ))
        
        fig_roc.update_layout(
            template="plotly_dark",
            title="ROC Curves — All Models",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_roc, use_container_width=True)
    
    # TAB 3: Confusion Matrix
    with tabs[2]:
        section_header("Confusion Matrix")
        
        selected_model = st.selectbox(
            "Select model:",
            list(models.keys()),
            format_func=lambda x: DISPLAY_NAMES.get(x, x)
        )
        
        y_pred = models[selected_model].predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        fig_cm = ff.create_annotated_heatmap(
            cm,
            x=["Predicted: No", "Predicted: Yes"],
            y=["Actual: No", "Actual: Yes"],
            colorscale="Reds",
            showscale=True
        )
        fig_cm.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_cm, use_container_width=True)
    
    # TAB 4: Feature Importance
    with tabs[3]:
        section_header("Feature Importance")
        
        selected_fi = st.selectbox(
            "Select model for features:",
            list(models.keys()),
            format_func=lambda x: DISPLAY_NAMES.get(x, x),
            key="fi_select"
        )
        
        try:
            fi_df = get_feature_importances(selected_fi)
            if fi_df is not None and not fi_df.empty:
                fig_fi = px.bar(
                    fi_df.head(15),
                    x="importance",
                    y="feature",
                    orientation="h",
                    template="plotly_dark",
                    color="importance",
                    color_continuous_scale="Reds",
                    title=f"Top 15 Features — {selected_fi}"
                )
                fig_fi.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(autorange="reversed")
                )
                st.plotly_chart(fig_fi, width="stretch")
            else:
                st.info("Feature importances not available for this model.")
        except Exception as e:
            st.warning(f"Could not compute feature importance: {e}")

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
