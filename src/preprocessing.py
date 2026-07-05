"""Shared preprocessing for the Cars24 selling-price model.

Decisions here are informed by notebooks/eda.ipynb:
- `model` has 3,233 unique values -> frequency-encoded rather than one-hot.
- `make` has 41 values -> one-hot encoded.
- `km_driven` has extreme outliers (data-entry errors, e.g. 3.8M km) -> winsorized.
- Remaining dummy columns are already 0/1 encoded and used as-is.
"""
from __future__ import annotations

import pandas as pd

TARGET = "selling_price"
KM_DRIVEN_CAP_QUANTILE = 0.99
DROP_COLUMNS = ["model"]


def load_data(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def fit_preprocessing_stats(df: pd.DataFrame) -> dict:
    """Compute stats from training data only, to avoid leakage into val/test."""
    return {
        "km_driven_cap": df["km_driven"].quantile(KM_DRIVEN_CAP_QUANTILE),
        "model_freq": df["model"].value_counts(normalize=True).to_dict(),
        "make_categories": sorted(df["make"].unique().tolist()),
    }


def transform(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Apply preprocessing using stats fitted on the training set."""
    df = df.copy()

    df["km_driven"] = df["km_driven"].clip(upper=stats["km_driven_cap"])

    default_freq = min(stats["model_freq"].values()) if stats["model_freq"] else 0.0
    df["model_freq"] = df["model"].map(stats["model_freq"]).fillna(default_freq)

    df = df.drop(columns=DROP_COLUMNS)

    make_dummies = pd.get_dummies(df["make"], prefix="make")
    make_dummies = make_dummies.reindex(
        columns=[f"make_{m}" for m in stats["make_categories"]], fill_value=0
    )
    df = pd.concat([df.drop(columns=["make"]), make_dummies], axis=1)

    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y
