"""Shared helpers for loading and summarizing the churn dataset."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "processed" / "churn_cleaned.csv"

# Standardized column names for the dataset
TARGET_COLUMN = "Churn Value"
NUMERIC_COLUMNS = ["Tenure Months", "Monthly Charges", "Total Charges"]
CATEGORICAL_COLUMNS = [
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Paperless Billing",
    "Payment Method",
]
FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS
VALID_CATEGORIES = {}


def load_data() -> pd.DataFrame:
    """Load and clean the processed churn dataset."""
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        st.error("⚠️ Dataset not found. Please ensure `data/processed/churn_cleaned.csv` exists.")
        st.stop()

    # Ensure target and numeric types
    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "Total Charges" in df.columns:
        df["Total Charges"] = df["Total Charges"].fillna(
            df.get("Monthly Charges", 0) * df.get("Tenure Months", 0)
        )
        df["Total Charges"] = df["Total Charges"].round(2)

    for column in CATEGORICAL_COLUMNS:
        if column in df.columns:
            df[column] = df[column].astype(str)

    return df


def get_valid_categories(df: pd.DataFrame) -> dict:
    """Return valid categories for all categorical columns."""
    return {col: sorted(df[col].dropna().unique().tolist()) for col in CATEGORICAL_COLUMNS if col in df.columns}


def validate_prediction_input(input_dict: dict, df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Validate user input for prediction."""
    errors: list[str] = []
    if input_dict.get("Tenure Months", 0) < 0:
        errors.append("Tenure must be >= 0")
    if input_dict.get("Monthly Charges", 0) < 0:
        errors.append("Monthly Charges must be >= 0")
    # Check categorical values
    valid = get_valid_categories(df)
    for k, v in input_dict.items():
        if k in valid and v not in valid[k]:
            errors.append(f"Invalid value for {k}: {v}")
    return (len(errors) == 0, errors)


def preprocess_for_prediction(input_dict: dict) -> pd.DataFrame:
    """Convert user input dict to a preprocessable DataFrame."""
    return pd.DataFrame([input_dict])[FEATURE_COLUMNS]


def get_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return summary of missing values."""
    missing = df.isna().sum()
    summary = (
        pd.DataFrame({"Column": missing.index, "Missing Values": missing.values})
        .assign(Missing_Percentage=lambda data: (data["Missing Values"] / len(df) * 100).round(2))
        .sort_values(["Missing Values", "Missing_Percentage"], ascending=False)
    )
    return summary[summary["Missing Values"] > 0]


def get_data_type_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return data type summary."""
    dtypes = df.dtypes.reset_index()
    dtypes.columns = ["Column", "Data Type"]
    dtypes["Data Type"] = dtypes["Data Type"].astype(str)
    return dtypes


def get_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    return df[NUMERIC_COLUMNS].describe().T.reset_index().rename(columns={"index": "Metric"})


def get_summary_stats(df: pd.DataFrame | None = None) -> dict:
    df = load_data() if df is None else df.copy()
    churn_counts = df[TARGET_COLUMN].value_counts().reindex([0, 1], fill_value=0)
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "missing_values": int(df.isna().sum().sum()),
        "churned": int(churn_counts.get(1, 0)),
        "retained": int(churn_counts.get(0, 0)),
        "churn_rate": round(df[TARGET_COLUMN].mean() * 100, 2),
        "numeric_columns": len(NUMERIC_COLUMNS),
        "categorical_columns": len(CATEGORICAL_COLUMNS),
    }


def get_churn_analysis_data(df: pd.DataFrame | None = None) -> dict:
    df = load_data() if df is None else df.copy()
    churn_col = df[TARGET_COLUMN].astype(int)

    tenure_bins = pd.cut(
        df.get("Tenure Months", pd.Series(dtype=float)),
        bins=[0, 12, 24, 36, 48, 60, 120],
        labels=["0-12", "13-24", "25-36", "37-48", "49-60", "60+"],
        include_lowest=True,
    )

    return {
        "contract_rate": (
            df.assign(churned=churn_col)
            .groupby("Contract")["churned"]
            .mean()
            .reset_index()
            .rename(columns={"churned": "churn_rate"})
        ),
        "tenure_rate": (
            df.assign(churned=churn_col, tenure_group=tenure_bins)
            .groupby("tenure_group")["churned"]
            .mean()
            .reset_index()
            .rename(columns={"churned": "churn_rate"})
        ),
        "payment_rate": (
            df.assign(churned=churn_col)
            .groupby("Payment Method")["churned"]
            .mean()
            .reset_index()
            .rename(columns={"churned": "churn_rate"})
        ),
        "internet_rate": (
            df.assign(churned=churn_col)
            .groupby("Internet Service")["churned"]
            .mean()
            .reset_index()
            .rename(columns={"churned": "churn_rate"})
        ),
    }


def preprocess_for_prediction(input_dict: dict) -> pd.DataFrame:
    df = load_data()
    defaults = {}

    for column in FEATURE_COLUMNS:
        if column in NUMERIC_COLUMNS:
            defaults[column] = float(df[column].median()) if column in df.columns else 0.0
        else:
            defaults[column] = str(df[column].mode().iloc[0]) if column in df.columns else ""

    payload = defaults.copy()
    for key, value in input_dict.items():
        payload[key] = value

    return pd.DataFrame([payload], columns=FEATURE_COLUMNS)


def get_project_summary(df: pd.DataFrame, best_model_name: str) -> dict:
    stats = get_summary_stats(df)
    return {
        "total_customers": stats["rows"],
        "churn_rate": stats["churn_rate"],
        "number_of_features": stats["columns"] - 1,
        "best_model": best_model_name,
    }
