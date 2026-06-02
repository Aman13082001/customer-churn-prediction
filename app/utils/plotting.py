"""Plotly chart helpers for the churn dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


CHURN_COLORS = {"Churned": "#ff6b6b", "Stayed": "#4ecdc4"}


def _dark_layout(title: str) -> dict:
    return {
        "title": title,
        "template": "plotly_dark",
        "paper_bgcolor": "#0f172a",
        "plot_bgcolor": "#0f172a",
        "font": {"color": "#e2e8f0"},
    }


def _churn_label_column(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["Churn Label"] = result["Churn Value"].map({1: "Churned", 0: "Stayed"})
    return result


def churn_distribution_chart(df: pd.DataFrame):
    plot_df = _churn_label_column(df)
    fig = px.pie(
        plot_df,
        names="Churn Label",
        title="Churn Distribution",
        hole=0.42,
        color="Churn Label",
        color_discrete_map=CHURN_COLORS,
    )
    fig.update_layout(**_dark_layout("Churn Distribution"))
    return fig


def class_distribution_chart(df: pd.DataFrame):
    plot_df = df["Churn Value"].value_counts().reindex([0, 1], fill_value=0)
    plot_df_reset = plot_df.reset_index()
    plot_df_reset.columns = ["Churn", "Count"]
    plot_df_reset["Churn Label"] = plot_df_reset["Churn"].map({0: "Stayed", 1: "Churned"})
    fig = px.bar(
        plot_df_reset,
        x="Churn Label",
        y="Count",
        color="Churn Label",
        color_discrete_map={"Stayed": "#4ecdc4", "Churned": "#ff6b6b"},
        title="Class Distribution",
    )
    fig.update_layout(**_dark_layout("Class Distribution"))
    return fig


def contract_churn_rate_chart(df: pd.DataFrame):
    plot_df = (
        df.assign(churned=df["Churn Value"].astype(int))
        .groupby("Contract", observed=True)["churned"]
        .mean()
        .reset_index()
        .rename(columns={"churned": "Churn Rate"})
    )
    fig = px.bar(
        plot_df,
        x="Contract",
        y="Churn Rate",
        color="Contract",
        title="Churn Rate by Contract Type",
    )
    fig.update_layout(**_dark_layout("Churn Rate by Contract Type"))
    return fig


def tenure_churn_rate_chart(df: pd.DataFrame):
    df = df.copy()
    df["Tenure Group"] = pd.cut(
        df["Tenure Months"],
        bins=[0, 12, 24, 36, 48, 60, 120],
        labels=["0-12", "13-24", "25-36", "37-48", "49-60", "60+"],
        include_lowest=True,
    )
    plot_df = (
        df.assign(churned=df["Churn Value"].astype(int))
        .groupby("Tenure Group", observed=True)["churned"]
        .mean()
        .reset_index()
        .rename(columns={"churned": "Churn Rate"})
    )
    fig = px.line(
        plot_df,
        x="Tenure Group",
        y="Churn Rate",
        markers=True,
        title="Churn Rate by Tenure Group",
    )
    fig.update_layout(**_dark_layout("Churn Rate by Tenure Group"))
    return fig


def monthly_charge_distribution_chart(df: pd.DataFrame):
    plot_df = _churn_label_column(df)
    fig = px.histogram(
        plot_df,
        x="Monthly Charges",
        color="Churn Label",
        barmode="overlay",
        nbins=40,
        opacity=0.7,
        title="Monthly Charges Distribution by Churn Status",
        color_discrete_map=CHURN_COLORS,
    )
    fig.update_layout(**_dark_layout("Monthly Charges Distribution by Churn Status"))
    return fig


def payment_method_churn_rate_chart(df: pd.DataFrame):
    plot_df = (
        df.assign(churned=df["Churn Value"].astype(int))
        .groupby("Payment Method", observed=True)["churned"]
        .mean()
        .reset_index()
        .rename(columns={"churned": "Churn Rate"})
    )
    fig = px.bar(
        plot_df,
        x="Payment Method",
        y="Churn Rate",
        color="Payment Method",
        title="Churn Rate by Payment Method",
    )
    fig.update_layout(**_dark_layout("Churn Rate by Payment Method"))
    return fig


def internet_service_churn_rate_chart(df: pd.DataFrame):
    plot_df = (
        df.assign(churned=df["Churn Value"].astype(int))
        .groupby("Internet Service", observed=True)["churned"]
        .mean()
        .reset_index()
        .rename(columns={"churned": "Churn Rate"})
    )
    fig = px.bar(
        plot_df,
        x="Internet Service",
        y="Churn Rate",
        color="Internet Service",
        title="Churn Rate by Internet Service",
    )
    fig.update_layout(**_dark_layout("Churn Rate by Internet Service"))
    return fig


def tenure_monthly_scatter_chart(df: pd.DataFrame):
    plot_df = _churn_label_column(df)
    fig = px.scatter(
        plot_df,
        x="Tenure Months",
        y="Monthly Charges",
        color="Churn Label",
        hover_data=["Contract", "Payment Method"],
        title="Tenure vs Monthly Charges",
        color_discrete_map=CHURN_COLORS,
    )
    fig.update_layout(**_dark_layout("Tenure vs Monthly Charges"))
    return fig


def correlation_heatmap(df: pd.DataFrame):
    numeric_df = df.select_dtypes(include=["number"])
    corr = numeric_df.corr()
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="Viridis",
        title="Correlation Heatmap of Numeric Features",
    )
    fig.update_layout(**_dark_layout("Correlation Heatmap of Numeric Features"))
    return fig


def feature_type_breakdown_chart(df: pd.DataFrame):
    counts = {
        "Categorical": sum(1 for column in df.columns if column != "Churn Value" and df[column].dtype == object),
        "Numeric": sum(1 for column in df.columns if column != "Churn Value" and df[column].dtype != object),
    }
    fig = px.bar(
        x=list(counts.keys()),
        y=list(counts.values()),
        color=list(counts.keys()),
        title="Feature Type Breakdown",
    )
    fig.update_layout(**_dark_layout("Feature Type Breakdown"))
    return fig


def contract_churn_chart(df: pd.DataFrame):
    return contract_churn_rate_chart(df)


def monthly_charges_chart(df: pd.DataFrame):
    return monthly_charge_distribution_chart(df)


def tenure_churn_chart(df: pd.DataFrame):
    return tenure_churn_rate_chart(df)


def payment_method_chart(df: pd.DataFrame):
    return payment_method_churn_rate_chart(df)


def dataset_distribution_chart(df: pd.DataFrame):
    return tenure_monthly_scatter_chart(df)


def model_comparison_chart(metrics_df: pd.DataFrame):
    long_df = metrics_df.melt(
        id_vars="model",
        value_vars=["accuracy", "precision", "recall", "f1", "roc_auc"],
        var_name="Metric",
        value_name="Score",
    )
    fig = px.bar(
        long_df,
        x="model",
        y="Score",
        color="Metric",
        barmode="group",
        title="Model Metric Comparison",
    )
    fig.update_layout(**_dark_layout("Model Metric Comparison"))
    return fig


def confusion_matrix_chart(confusion_matrix_values, labels, model_name):
    fig = go.Figure(
        data=go.Heatmap(
            z=confusion_matrix_values,
            x=labels,
            y=labels,
            text=confusion_matrix_values,
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=True,
        )
    )
    fig.update_layout(
        **_dark_layout(f"Confusion Matrix — {model_name}"),
        xaxis_title="Predicted",
        yaxis_title="Actual",
    )
    return fig


def roc_curve_chart(roc_data: dict[str, dict[str, float]]):
    fig = go.Figure()
    for model_name, payload in roc_data.items():
        fig.add_trace(
            go.Scatter(
                x=payload["fpr"],
                y=payload["tpr"],
                mode="lines",
                name=f"{model_name} (AUC = {payload['roc_auc']:.4f})",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash"),
            name="Random Baseline",
        )
    )
    fig.update_layout(
        **_dark_layout("ROC Curve Comparison"),
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
    )
    return fig


def feature_importance_chart(importances: pd.DataFrame, model_name: str):
    fig = px.bar(
        importances.sort_values("importance", ascending=False),
        x="importance",
        y="feature",
        orientation="h",
        title=f"Top 10 Feature Importances — {model_name}",
    )
    fig.update_layout(**_dark_layout(f"Top 10 Feature Importances — {model_name}"))
    return fig


def probability_gauge_chart(probability: float):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            title={"text": "Predicted Churn Probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#ff6b6b"},
                "steps": [
                    {"range": [0, 40], "color": "#4ecdc4"},
                    {"range": [40, 70], "color": "#f59e0b"},
                    {"range": [70, 100], "color": "#ff6b6b"},
                ],
            },
        )
    )
    fig.update_layout(**_dark_layout("Predicted Churn Probability"))
    return fig
