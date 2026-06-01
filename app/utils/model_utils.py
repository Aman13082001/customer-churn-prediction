"""Model loading, evaluation, and prediction helpers."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
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
    "Decision Tree": "decision_tree.joblib",
    "Random Forest": "random_forest.joblib",
    "XGBoost": "xgboost.joblib",
}


def load_all_models() -> dict[str, object]:
    """Load all trained model artifacts from disk."""
    models = {}
    for model_name, filename in MODEL_FILES.items():
        artifact_path = MODELS_DIR / filename
        if not artifact_path.exists():
            raise FileNotFoundError(f"Missing model artifact: {artifact_path}")
        models[model_name] = joblib.load(artifact_path)
    return models


def load_metadata() -> dict:
    """Load saved metadata for model prediction and feature ordering."""
    try:
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def get_evaluation_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a train/test split for model evaluation."""
    df = load_data()
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN].astype(int)
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def evaluate_all_models(X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """Evaluate all saved models on a test split and return metrics."""
    models = load_all_models()
    results = []

    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        results.append(
            {
                "model": model_name,
                "accuracy": round(accuracy_score(y_test, y_pred), 4),
                "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
                "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
                "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
                "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
            }
        )

    return pd.DataFrame(results).sort_values("roc_auc", ascending=False)


def get_feature_importances(model_name: str) -> pd.DataFrame | None:
    """Return top-10 feature importances for tree-based models."""
    model = load_all_models()[model_name]
    estimator = model.named_steps.get("model", model)

    if not hasattr(estimator, "feature_importances_"):
        return None

    feature_names = model.named_steps["preprocess"].get_feature_names_out()
    importances = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": estimator.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    return importances.head(10)


def get_best_model_name(metrics: pd.DataFrame) -> str:
    """Return the model name with the highest ROC-AUC score."""
    return str(metrics.loc[metrics["roc_auc"].idxmax(), "model"])


def get_risk_label(probability: float) -> str:
    """Convert churn probability into a business-friendly risk label."""
    if probability >= 0.70:
        return "High Risk"
    if probability >= 0.40:
        return "Medium Risk"
    return "Low Risk"


def get_retention_recommendations(input_features: pd.DataFrame, probability: float) -> list[str]:
    """Generate a ranked list of retention recommendations for a customer profile."""
    row = input_features.iloc[0]
    recommendations = []

    if probability >= 0.70 or row["Contract"] == "Month-to-month":
        recommendations.append("Reach out with a retention offer and proactive support.")
    if row["Tenure Months"] < 12:
        recommendations.append("Send an onboarding and engagement campaign to improve early stickiness.")
    if row["Internet Service"] == "Fiber optic":
        recommendations.append("Prioritize network reliability messaging and support follow-up.")
    if row["Monthly Charges"] > load_data()["Monthly Charges"].median():
        recommendations.append("Review pricing sensitivity and offer a plan adjustment suggestion.")
    if row["Paperless Billing"] == "Yes":
        recommendations.append("Use a personalized digital engagement workflow to reduce churn risk.")

    if not recommendations:
        recommendations.append("Continue standard relationship management with periodic usage nudges.")

    return recommendations[:3]


def predict_single(model_name: str, input_features: dict | pd.DataFrame) -> dict:
    """Predict churn probability for a single profile using a selected model."""
    models = load_all_models()
    model = models[model_name]

    if isinstance(input_features, dict):
        input_df = preprocess_for_prediction(input_features)
    else:
        input_df = input_features.copy()

    probability = float(model.predict_proba(input_df)[0, 1])
    prediction = int(model.predict(input_df)[0])
    return {
        "model": model_name,
        "prediction": prediction,
        "probability": probability,
        "risk_label": get_risk_label(probability),
        "confidence": round(max(probability, 1 - probability), 4),
    }


def get_confusion_matrix(model_name: str) -> tuple[list[list[int]], pd.DataFrame, pd.DataFrame]:
    """Return the confusion matrix for the selected model plus the evaluation split."""
    X_train, X_test, y_train, y_test = get_evaluation_split()
    model = load_all_models()[model_name]
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred).tolist()
    return cm, X_test, y_test


def get_model_status() -> dict:
    """Return which saved models are available."""
    models = load_all_models()
    return {name: True for name in models.keys()}


def get_business_recommendation(probability: float) -> str:
    """Backward-compatible recommendation text for existing callers."""
    return get_retention_recommendations(
        preprocess_for_prediction(
            {
                "Gender": "Male",
                "Senior Citizen": "No",
                "Partner": "No",
                "Dependents": "No",
                "Tenure Months": 12,
                "Phone Service": "Yes",
                "Multiple Lines": "No",
                "Internet Service": "DSL",
                "Online Security": "No",
                "Online Backup": "No",
                "Device Protection": "No",
                "Tech Support": "No",
                "Streaming TV": "No",
                "Streaming Movies": "No",
                "Contract": "Month-to-month",
                "Paperless Billing": "Yes",
                "Payment Method": "Electronic check",
                "Monthly Charges": 70.0,
                "Total Charges": 500.0,
            }
        ),
        probability,
    )[0]


def build_prediction_payload(form_inputs: dict, metadata: dict | None = None) -> pd.DataFrame:
    """Backward-compatible helper returning a model-ready payload."""
    return preprocess_for_prediction(form_inputs)


def predict_customer(form_inputs: dict) -> dict:
    """Backward-compatible wrapper for the prediction form."""
    result = predict_single("XGBoost", form_inputs)
    return {
        "best_model": result["model"],
        "prediction": result["prediction"],
        "probability": result["probability"],
        "recommendation": get_business_recommendation(result["probability"]),
    }
