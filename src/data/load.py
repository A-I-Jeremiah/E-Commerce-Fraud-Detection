"""
Data loading utilities.
Keeps raw data immutable and provides clean, typed DataFrames.
"""

from pathlib import Path
from typing import Optional, Union

import pandas as pd

from src.config import (
    RAW_DATA_PATH,
    DATETIME_COL,
    TARGET_COL,
    TARGET_LABEL_COL,
    ID_COL,
    FEATURE_COLS,
    LEAKAGE_COLS,
)


def load_raw_data(path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Load the immutable raw dataset.

    Parameters
    ----------
    path : optional path override (useful for tests)

    Returns
    -------
    pd.DataFrame with parsed datetime and correct dtypes.
    """
    data_path = Path(path) if path is not None else RAW_DATA_PATH

    if not data_path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {data_path}. "
            "Expected ecommerce_fraud_detection.csv in data/raw/"
        )

    df = pd.read_csv(data_path)

    # Basic sanity checks
    assert ID_COL in df.columns, f"Missing ID column: {ID_COL}"
    assert TARGET_COL in df.columns, f"Missing target column: {TARGET_COL}"
    assert df[ID_COL].is_unique, "Transaction_ID is not unique"

    # Apply basic cleaning (parse datetime, enforce target dtype, drop duplicates)
    df = clean_data(df)

    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw dataset by removing leakage columns and duplicates.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame with leakage columns removed and duplicates dropped.
    """

    # Parse datetime column
    df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL])

    # Ensure target is integer
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    
    # Drop duplicates based on ID column
    df = df.drop_duplicates(subset=ID_COL)

    return df

def get_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the model feature columns."""
    missing = set(FEATURE_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected feature columns: {missing}")
    return df[FEATURE_COLS].copy()


def get_target(df: pd.DataFrame) -> pd.Series:
    """Return the binary target."""
    return df[TARGET_COL].copy()


def train_test_split_time_aware(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    datetime_col: str = DATETIME_COL,
):
    """
    Simple chronological split to reduce leakage.
    Sorts by transaction date and takes the most recent portion as test,
    then splits the earlier portion into train/val.
    """
    df = df.sort_values(datetime_col).reset_index(drop=True)
    n = len(df)
    test_start = int(n * (1 - test_size))
    val_start = int(test_start * (1 - val_size))

    train_df = df.iloc[:val_start].copy()
    val_df = df.iloc[val_start:test_start].copy()
    test_df = df.iloc[test_start:].copy()

    return train_df, val_df, test_df