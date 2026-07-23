"""
utils/data_loader.py
====================
Responsible for reading the raw CSV into a pandas DataFrame and performing
light, defensive cleaning so the rest of the app can assume tidy data.
"""

from __future__ import annotations

import os
import pandas as pd

import config


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the expected dataset CSV is missing."""


def load_data(path: str | None = None) -> pd.DataFrame:
    """Load the social-media dataset.

    Parameters
    ----------
    path:
        Optional override for the CSV location. Defaults to
        ``config.DATASET_PATH``.

    Returns
    -------
    pandas.DataFrame
        A cleaned DataFrame ready for EDA / modelling.

    Raises
    ------
    DatasetNotFoundError
        If the CSV cannot be found. The message hints at how to create it.
    """
    path = path or config.DATASET_PATH
    if not os.path.exists(path):
        raise DatasetNotFoundError(
            f"Dataset not found at '{path}'. "
            "Run `python generate_dataset.py` first to create it."
        )

    df = pd.read_csv(path)
    return _clean(df)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply defensive cleaning: drop dupes, coerce types, drop bad rows."""
    df = df.drop_duplicates().copy()

    # Ensure numeric columns really are numeric; coerce errors to NaN.
    numeric_cols = config.NUMERIC_FEATURES + [
        "likes",
        "comments",
        "shares",
        config.TARGET,
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Strip whitespace on categorical columns so joins/lookups are reliable.
    for col in config.CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Drop rows missing the target or any model feature.
    required = config.CATEGORICAL_FEATURES + config.NUMERIC_FEATURES + [config.TARGET]
    df = df.dropna(subset=[c for c in required if c in df.columns])

    return df.reset_index(drop=True)


def dataset_summary(df: pd.DataFrame) -> dict:
    """Return a small dict of headline numbers for the dashboard hero cards."""
    return {
        "total_posts": int(len(df)),
        "avg_engagement": round(float(df[config.TARGET].mean()), 2),
        "max_engagement": round(float(df[config.TARGET].max()), 2),
        "platforms": int(df["platform"].nunique()),
        "total_likes": int(df["likes"].sum()) if "likes" in df else 0,
        "best_platform": (
            df.groupby("platform")[config.TARGET].mean().idxmax()
            if "platform" in df
            else "N/A"
        ),
    }
