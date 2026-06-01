"""Feature engineering functions."""

from __future__ import annotations

import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature transformations and return transformed DataFrame."""
    transformed = df.copy()

    numeric_columns = [col for col in transformed.columns if transformed[col].dtype.kind in "biufc"]
    for column in numeric_columns:
        transformed[column] = pd.to_numeric(transformed[column], errors="coerce")
        transformed[column] = transformed[column].fillna(transformed[column].median())

    categorical_columns = transformed.select_dtypes(include="object").columns.tolist()
    for column in categorical_columns:
        transformed[column] = transformed[column].astype(str).str.strip().fillna("Unknown")

    if "Total Charges" in transformed.columns and "Monthly Charges" in transformed.columns:
        transformed["Total Charges"] = transformed["Total Charges"].replace({"": None}).astype(float)
        transformed["Total Charges"] = transformed["Total Charges"].fillna(
            transformed["Monthly Charges"].fillna(0) * transformed["Tenure Months"].fillna(0)
        )
        transformed["Total Charges"] = transformed["Total Charges"].round(2)

    return transformed
