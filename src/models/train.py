"""Model training script."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from app.utils.data_utils import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
    load_data,
)
from src.features.feature_engineering import engineer_features

MODEL_DIR = ROOT_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
METADATA_PATH = MODEL_DIR / "model_metadata.json"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "Random Forest": "random_forest.joblib",
    "XGBoost": "xgboost.joblib",
}


def get_preprocessor() -> ColumnTransformer:
    """Create a preprocessing transformer for categorical and numeric features."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_COLUMNS),
            ("cat", categorical_transformer, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
        sparse_threshold=0,
    )


def build_model_pipeline(model) -> Pipeline:
    """Create a full pipeline for preprocessing and model training."""
    return Pipeline(
        steps=[
            ("preprocess", get_preprocessor()),
            ("model", model),
        ]
    )


def train_models() -> dict[str, float]:
    """Train all candidate models and return evaluation scores."""
    df = load_data()
    df = engineer_features(df)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidate_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42),
    }

    evaluation_scores: dict[str, float] = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for model_name, estimator in candidate_models.items():
        pipeline = build_model_pipeline(estimator)
        pipeline.fit(X_train, y_train)

        if hasattr(pipeline, "predict_proba"):
            score = pipeline.score(X_test, y_test)
        else:
            score = pipeline.score(X_test, y_test)

        artifact_path = MODEL_DIR / MODEL_FILES[model_name]
        joblib.dump(pipeline, artifact_path)
        evaluation_scores[model_name] = round(score, 4)

        print(f"Saved {model_name} pipeline to {artifact_path}")

    metadata = {
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "numeric_columns": NUMERIC_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "model_files": MODEL_FILES,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved metadata to {METADATA_PATH}")

    cv_scores = {}
    for model_name, estimator in candidate_models.items():
        pipeline = build_model_pipeline(estimator)
        # Run cross-validation only on the training split to avoid data leakage
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")
        cv_scores[model_name] = round(float(scores.mean()), 4)

    print("Cross-validation accuracy scores:")
    for model_name, score in cv_scores.items():
        print(f" - {model_name}: {score}")

    return evaluation_scores


if __name__ == "__main__":
    results = train_models()
    print("Training complete. Evaluation scores:")
    for model_name, score in results.items():
        print(f" - {model_name}: {score}")
