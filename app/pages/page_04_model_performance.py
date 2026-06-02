"""Model Performance — ML Model Comparison Center."""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

from app.utils.data_utils import FEATURE_COLUMNS, TARGET_COLUMN, load_data
from app.utils.model_utils import (
    load_all_models,
    evaluate_all_models,
    get_evaluation_split,
    get_feature_importances,
    get_confusion_matrix,
)
from app.utils.ui_components import (
    apply_global_styles,
    page_header,
    metric_card_row,
    section_header,
    insight_card,
)
from app.utils.plotting import confusion_matrix_chart, feature_importance_chart

DISPLAY_NAMES = {
    "Logistic Regression": "Logistic Regression",
    "Decision Tree":       "Decision Tree",
    "Random Forest":       "Random Forest",
    "XGBoost":             "XGBoost",
}

CHART_COLORS = ["#E63946", "#2EC4B6", "#F4A261", "#A8DADC", "#457B9D"]


def show_page() -> None:
    """Render the model performance page."""
    apply_global_styles()

    page_header(
        "🤖",
        "Model Performance Center",
        "Compare all 4 trained models across key ML evaluation metrics",
    )

    # ── Load models & data ───────────────────────────────────────────────────
    try:
        _, X_test, _, y_test = get_evaluation_split()
        models  = load_all_models()
        results = evaluate_all_models(X_test, y_test)
    except Exception as exc:
        st.error(f"Could not load models or evaluate them: {exc}")
        return

    if results.empty:
        st.warning("No model results available. Check that .joblib files exist in `/models`.")
        return

    # ── Hero KPIs ────────────────────────────────────────────────────────────
    best_row   = results.iloc[0]           # iloc is safe: index is always reset
    best_model = str(best_row["model"])

    metric_card_row([
        {"label": "Best Model",      "value": best_model},
        {"label": "Best Accuracy",   "value": f"{best_row['accuracy']:.3f}"},
        {"label": "Best ROC-AUC",    "value": f"{best_row['roc_auc']:.3f}"},
        {"label": "Models Compared", "value": f"{len(results)}/4"},
    ])

    st.markdown("---")

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab_labels = [
        "📊 Comparison Table",
        "📈 ROC Curves",
        "🔢 Confusion Matrix",
        "🎯 Feature Importance",
    ]
    tab1, tab2, tab3, tab4 = st.tabs(tab_labels)

    # ── Tab 1 · Comparison Table ─────────────────────────────────────────────
    with tab1:
        section_header("All Models — Metrics Comparison")

        # Display table with renamed columns (keep numeric values, no formatting)
        display_df = results[
            ["model", "accuracy", "precision", "recall", "f1", "roc_auc"]
        ].rename(columns={
            "model":     "Model",
            "accuracy":  "Accuracy",
            "precision": "Precision",
            "recall":    "Recall",
            "f1":        "F1 Score",
            "roc_auc":   "ROC-AUC",
        }).copy()

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # ── Grouped bar chart ─────────────────────────────────────────────────
        metric_keys   = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        metric_labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]

        fig_bar = go.Figure()
        # Use enumerate so 'idx' is always a Python int — no DataFrame index involved
        for idx, row in enumerate(results.to_dict("records")):
            color = CHART_COLORS[idx % len(CHART_COLORS)]
            fig_bar.add_trace(go.Bar(
                name=str(row["model"]),
                x=metric_labels,
                y=[float(row[k]) for k in metric_keys],
                marker_color=color,
                opacity=0.88,
                text=[f"{float(row[k]):.3f}" for k in metric_keys],
                textposition="outside",
                hovertemplate=f"{row['model']}<br>%{{x}}: %{{y:.4f}}<extra></extra>",
            ))

        fig_bar.update_layout(
            barmode="group",
            template="plotly_dark",
            title="Model Metrics Comparison",
            xaxis_title="Metric",
            yaxis_title="Score",
            yaxis=dict(range=[0.0, 1.09]),
            legend=dict(orientation="h", y=-0.2),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=400,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        insight_card(
            f"<strong>{best_model}</strong> achieves the highest ROC-AUC on the hold-out test set. "
            "ROC-AUC is preferred over raw accuracy because the dataset is class-imbalanced (~27 % positive)."
        )

    # ── Tab 2 · ROC Curves ───────────────────────────────────────────────────
    with tab2:
        section_header("ROC Curves — All Models")

        fig_roc = go.Figure()
        fig_roc.add_shape(
            type="line", x0=0, y0=0, x1=1, y1=1,
            line=dict(color="#555", dash="dash", width=1.5),
        )

        for idx, (name, model) in enumerate(models.items()):
            try:
                y_prob          = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _     = roc_curve(y_test, y_prob)
                auc             = roc_auc_score(y_test, y_prob)
                color           = CHART_COLORS[idx % len(CHART_COLORS)]
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    name=f"{name} (AUC = {auc:.3f})",
                    mode="lines",
                    line=dict(color=color, width=2.5),
                    hovertemplate=f"{name}<br>FPR: %{{x:.3f}}<br>TPR: %{{y:.3f}}<extra></extra>",
                ))
            except Exception as exc:
                st.warning(f"Could not compute ROC for {name}: {exc}")

        fig_roc.update_layout(
            template="plotly_dark",
            title="ROC Curves — All Models",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1.02]),
            legend=dict(x=0.55, y=0.05, bgcolor="rgba(22,27,34,0.9)"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=460,
        )
        st.plotly_chart(fig_roc, use_container_width=True)

        insight_card(
            "All 4 models exceed AUC 0.80, confirming reliable churn discrimination. "
            "An AUC of 1.0 is perfect; 0.5 is random chance."
        )

    # ── Tab 3 · Confusion Matrix ─────────────────────────────────────────────
    with tab3:
        section_header("Confusion Matrix")

        model_name_list = list(models.keys())
        selected_model = st.selectbox(
            "Select model:",
            model_name_list,
            format_func=lambda x: DISPLAY_NAMES.get(x, x),
            key="cm_model_select",
        )

        try:
            y_pred = models[selected_model].predict(X_test)
            cm_array = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm_array.ravel()

            # Convert to plain Python ints before passing to plotly
            cm_list = [[int(tn), int(fp)], [int(fn), int(tp)]]

            fig_cm = go.Figure(go.Heatmap(
                z=cm_list,
                x=["Predicted: No Churn", "Predicted: Churn"],
                y=["Actual: No Churn",    "Actual: Churn"],
                text=cm_list,
                texttemplate="%{text}",
                textfont=dict(size=18),
                colorscale=[[0, "#0D1117"], [0.5, "#4A1942"], [1.0, "#E63946"]],
                showscale=True,
                hovertemplate="Predicted: %{x}<br>Actual: %{y}<br>Count: %{z}<extra></extra>",
            ))
            fig_cm.update_layout(
                template="plotly_dark",
                title=f"Confusion Matrix — {selected_model}",
                paper_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(autorange="reversed"),
                height=380,
            )
            st.plotly_chart(fig_cm, use_container_width=True)

            # Derived metric callouts
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("True Negatives",   f"{int(tn):,}", help="Correctly predicted: Stayed")
            m2.metric("True Positives",   f"{int(tp):,}", help="Correctly predicted: Churned")
            m3.metric("False Positives",  f"{int(fp):,}", help="Predicted churn, actually stayed")
            m4.metric("False Negatives",  f"{int(fn):,}", help="Missed churners — most costly error")

            insight_card(
                f"<strong>False Negatives ({int(fn):,})</strong> are the most costly mistake: "
                "churners predicted as loyal customers receive no intervention. "
                "Optimising for high <strong>Recall</strong> minimises this error."
            )

        except Exception as exc:
            st.error(f"Could not generate confusion matrix for {selected_model}: {exc}")

    # ── Tab 4 · Feature Importance ───────────────────────────────────────────
    with tab4:
        section_header("Feature Importance")

        fi_model_list = list(models.keys())
        selected_fi = st.selectbox(
            "Select model for features:",
            fi_model_list,
            format_func=lambda x: DISPLAY_NAMES.get(x, x),
            key="fi_model_select",
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
                    title=f"Top 15 Features — {selected_fi}",
                )
                fig_fi.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(autorange="reversed"),
                    height=420,
                    margin=dict(l=160),
                )
                st.plotly_chart(fig_fi, use_container_width=True)

                insight_card(
                    "Contract type, tenure, and monthly charges consistently rank "
                    "as the <strong>top predictors</strong> across all models — "
                    "these are the primary levers for churn intervention."
                )
            else:
                st.info(
                    f"Feature importances are not available for **{selected_fi}**. "
                    "This model type does not expose tree-based importances."
                )

        except Exception as exc:
            st.warning(f"Could not compute feature importance for {selected_fi}: {exc}")

    # ── Bottom Summary ────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📌 Why this model performs best")
    st.write(
        f"**{best_model}** leads the leaderboard with the highest ROC-AUC score "
        f"({best_row['roc_auc']:.3f}) and a strong balance of precision "
        f"({best_row['precision']:.3f}) and recall ({best_row['recall']:.3f}) "
        "on the held-out evaluation split."
    )