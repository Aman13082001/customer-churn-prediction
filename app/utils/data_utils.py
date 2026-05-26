"""Shared helpers for loading and summarizing the churn dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "processed" / "churn_cleaned.csv"
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


def load_data() -> pd.DataFrame:
    """Load and clean the processed churn dataset."""
    df = pd.read_csv(DATA_PATH)
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["Total Charges"] = df["Total Charges"].fillna(
        df["Monthly Charges"].fillna(0) * df["Tenure Months"].fillna(0)
    )
    df["Total Charges"] = df["Total Charges"].round(2)

    for column in CATEGORICAL_COLUMNS:
        df[column] = df[column].astype(str)

    return df


load_processed_data = load_data


def get_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return missing values information with percentage."""
    missing = df.isna().sum()
    summary = (
        pd.DataFrame({"Column": missing.index, "Missing Values": missing.values})
        .assign(Missing_Percentage=lambda data: (data["Missing Values"] / len(df) * 100).round(2))
        .sort_values(["Missing Values", "Missing_Percentage"], ascending=False)
    )
    return summary[summary["Missing Values"] > 0]


def get_data_type_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return the dataset dtypes."""
    dtypes = df.dtypes.reset_index()
    dtypes.columns = ["Column", "Data Type"]
    return dtypes


def get_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for numeric columns."""
    return df[NUMERIC_COLUMNS].describe().T.reset_index().rename(columns={"index": "Metric"})


def get_summary_stats(df: pd.DataFrame | None = None) -> dict:
    """Return key summary statistics for the dataset."""
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
    """Prepare aggregate churn analysis tables for the dashboard."""
    df = load_data() if df is None else df.copy()
    churn_col = df[TARGET_COLUMN].astype(int)

    tenure_bins = pd.cut(
        df["Tenure Months"],
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
    """Convert user form inputs into a model-ready feature frame."""
    df = load_data()
    defaults = {}

    for column in FEATURE_COLUMNS:
        if column in NUMERIC_COLUMNS:
            defaults[column] = float(df[column].median())
        else:
            defaults[column] = str(df[column].mode().iloc[0])

    payload = defaults.copy()
    for key, value in input_dict.items():
        payload[key] = value

    return pd.DataFrame([payload], columns=FEATURE_COLUMNS)


def get_project_summary(df: pd.DataFrame, best_model_name: str) -> dict:
    """Compute summary stats shown on the home page."""
    stats = get_summary_stats(df)
    return {
        "total_customers": stats["rows"],
        "churn_rate": stats["churn_rate"],
        "number_of_features": stats["columns"] - 1,
        "best_model": best_model_name,
    }