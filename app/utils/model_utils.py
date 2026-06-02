"""Model loading, evaluation, and prediction helpers."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from app.utils.data_utils import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_data,
    preprocess_for_prediction,
)

BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)
METADATA_PATH = MODELS_DIR / "model_metadata.json"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree":       "decision_tree.joblib",
    "Random Forest":       "random_forest.joblib",
    "XGBoost":             "xgboost.joblib",
}


@st.cache_resource
def load_all_models() -> dict[str, object]:
    """Load all trained model artifacts from disk."""
    models = {}
    for model_name, filename in MODEL_FILES.items():
        artifact_path = MODELS_DIR / filename
        if not artifact_path.exists():
            st.warning(f"⚠️ Missing model artifact: {filename}")
            continue
        try:
            models[model_name] = joblib.load(artifact_path)
        except Exception as exc:
            st.warning(f"⚠️ Could not load {model_name}: {exc}")
    return models


@st.cache_data
def load_metadata() -> dict:
    """Load saved metadata for model prediction and feature ordering."""
    try:
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def get_evaluation_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a reproducible train/test split for model evaluation."""
    df = load_data()
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN].astype(int)
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def evaluate_all_models(X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """
    Evaluate all saved models on a test split and return metrics DataFrame.
    Index is always reset to 0,1,2,3 — safe for iloc and positional operations.
    """
    models = load_all_models()
    results = []

    for model_name, model in models.items():
        try:
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            results.append({
                "model":     model_name,
                "accuracy":  round(float(accuracy_score(y_test, y_pred)),                      4),
                "precision": round(float(precision_score(y_test, y_pred, zero_division=0)),    4),
                "recall":    round(float(recall_score(y_test, y_pred,    zero_division=0)),    4),
                "f1":        round(float(f1_score(y_test, y_pred,        zero_division=0)),    4),
                "roc_auc":   round(float(roc_auc_score(y_test, y_prob)),                       4),
            })
        except Exception as exc:
            st.warning(f"Could not evaluate {model_name}: {exc}")

    df_results = (
        pd.DataFrame(results)
        .sort_values("roc_auc", ascending=False)
        .reset_index(drop=True)          # ← critical fix: always clean 0-based index
    )
    return df_results


def get_feature_importances(model_name: str) -> pd.DataFrame | None:
    """
    Return feature importances for tree-based models, or coefficients for
    linear models. Returns None if not supported.
    """
    models = load_all_models()
    if model_name not in models:
        return None

    model = models[model_name]

    # Try to find the final estimator step
    estimator = None
    if hasattr(model, "named_steps"):
        for step_key in ("model", "classifier", "estimator", "clf"):
            if step_key in model.named_steps:
                estimator = model.named_steps[step_key]
                break
        if estimator is None:
            estimator = list(model.named_steps.values())[-1]
    else:
        estimator = model

    # Determine importance values
    if hasattr(estimator, "feature_importances_"):
        raw_importances = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        raw_importances = np.abs(estimator.coef_[0])
    else:
        return None

    # Try to get transformed feature names from the preprocessor step
    feature_names: list[str] = list(FEATURE_COLUMNS)   # fallback
    if hasattr(model, "named_steps"):
        for step_key in ("preprocess", "preprocessor", "transformer", "prep"):
            if step_key in model.named_steps:
                try:
                    feature_names = list(
                        model.named_steps[step_key].get_feature_names_out()
                    )
                except Exception:
                    pass
                break

    # Align lengths (one-hot encoding can produce more features than raw columns)
    n = min(len(feature_names), len(raw_importances))
    df_imp = pd.DataFrame({
        "feature":    feature_names[:n],
        "importance": raw_importances[:n].tolist(),   # plain Python floats
    })
    df_imp = df_imp[df_imp["importance"] > 0]
    df_imp.sort_values("importance", ascending=False, inplace=True)
    df_imp.reset_index(drop=True, inplace=True)
    return df_imp.head(15)


def get_best_model_name(metrics: pd.DataFrame) -> str:
    """Return the model name with the highest ROC-AUC score."""
    if metrics.empty:
        return "N/A"
    return str(metrics.loc[metrics["roc_auc"].idxmax(), "model"])


def get_risk_label(probability: float) -> str:
    """Convert churn probability into a business-friendly risk label."""
    if probability >= 0.70:
        return "High Risk"
    if probability >= 0.40:
        return "Medium Risk"
    return "Low Risk"


def get_retention_recommendations(
    input_features: pd.DataFrame, probability: float
) -> list[str]:
    """Generate a ranked list of retention recommendations for a customer profile."""
    try:
        row = input_features.iloc[0]
        recommendations: list[str] = []

        if probability >= 0.70 or str(row.get("Contract", "")) == "Month-to-month":
            recommendations.append(
                "🎯 Reach out immediately with a personalised retention offer and proactive support."
            )
        tenure_val = row.get("Tenure Months", row.get("tenure", 99))
        if float(tenure_val) < 12:
            recommendations.append(
                "📧 Send an onboarding and early engagement campaign to improve first-year stickiness."
            )
        if str(row.get("Internet Service", "")) == "Fiber optic":
            recommendations.append(
                "📡 Prioritise network reliability messaging and a dedicated support follow-up."
            )
        df_ref = load_data()
        monthly_col = "Monthly Charges" if "Monthly Charges" in df_ref.columns else "MonthlyCharges"
        charge_col  = "Monthly Charges" if "Monthly Charges" in row.index else "MonthlyCharges"
        median_charge = float(df_ref[monthly_col].median())
        if float(row.get(charge_col, 0)) > median_charge:
            recommendations.append(
                "💰 Review pricing sensitivity and offer a plan-adjustment or bundle discount."
            )
        paperless_col = "Paperless Billing" if "Paperless Billing" in row.index else "PaperlessBilling"
        if str(row.get(paperless_col, "No")) == "Yes":
            recommendations.append(
                "📲 Use a personalised digital engagement workflow to reduce churn risk."
            )
        if not recommendations:
            recommendations.append(
                "✅ Continue standard relationship management with periodic usage nudges."
            )
        return recommendations[:4]
    except Exception:
        return ["✅ Continue standard relationship management."]


def predict_single(model_name: str, input_features: dict | pd.DataFrame) -> dict:
    """Predict churn probability for a single profile using a selected model."""
    models = load_all_models()
    model  = models[model_name]

    if isinstance(input_features, dict):
        input_df = preprocess_for_prediction(input_features)
    else:
        input_df = input_features.copy()

    probability = float(model.predict_proba(input_df)[0, 1])
    prediction  = int(model.predict(input_df)[0])
    return {
        "model":       model_name,
        "prediction":  prediction,
        "probability": probability,
        "risk_label":  get_risk_label(probability),
        "confidence":  round(max(probability, 1 - probability), 4),
    }


def get_confusion_matrix(
    model_name: str,
) -> tuple[list[list[int]], pd.DataFrame, pd.Series]:
    """Return the confusion matrix (as plain Python list) for the selected model."""
    _, X_test, _, y_test = get_evaluation_split()
    model  = load_all_models()[model_name]
    y_pred = model.predict(X_test)
    cm     = confusion_matrix(y_test, y_pred).tolist()   # plain Python ints
    return cm, X_test, y_test


def get_model_status() -> dict:
    """Return which saved models are available."""
    return {name: True for name in load_all_models().keys()}


# ── Backward-compatible helpers ───────────────────────────────────────────────

def get_business_recommendation(probability: float) -> str:
    dummy = preprocess_for_prediction({
        "Gender": "Male", "Senior Citizen": "No", "Partner": "No",
        "Dependents": "No", "Tenure Months": 12, "Phone Service": "Yes",
        "Multiple Lines": "No", "Internet Service": "DSL",
        "Online Security": "No", "Online Backup": "No",
        "Device Protection": "No", "Tech Support": "No",
        "Streaming TV": "No", "Streaming Movies": "No",
        "Contract": "Month-to-month", "Paperless Billing": "Yes",
        "Payment Method": "Electronic check",
        "Monthly Charges": 70.0, "Total Charges": 500.0,
    })
    return get_retention_recommendations(dummy, probability)[0]


def build_prediction_payload(
    form_inputs: dict, metadata: dict | None = None
) -> pd.DataFrame:
    return preprocess_for_prediction(form_inputs)


def predict_customer(form_inputs: dict) -> dict:
    result = predict_single("XGBoost", form_inputs)
    return {
        "best_model":     result["model"],
        "prediction":     result["prediction"],
        "probability":    result["probability"],
        "recommendation": get_business_recommendation(result["probability"]),
    }