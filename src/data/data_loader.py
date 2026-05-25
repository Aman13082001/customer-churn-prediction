"""Data loading utilities for the project."""

import pandas as pd


def load_raw_data(path: str) -> pd.DataFrame:
    """Load raw CSV data from `path`."""
    return pd.read_csv(path)
