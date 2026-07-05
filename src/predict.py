"""Load the saved artifacts and predict selling_price for new rows.

Example:
    python3 predict.py
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from preprocessing import split_features_target, transform

ARTIFACTS_DIR = Path("../artifacts")


def load_artifacts():
    model = joblib.load(ARTIFACTS_DIR / "model.joblib")
    stats = joblib.load(ARTIFACTS_DIR / "preprocessing_stats.joblib")
    feature_columns = joblib.load(ARTIFACTS_DIR / "feature_columns.joblib")
    return model, stats, feature_columns


def predict(df: pd.DataFrame) -> pd.Series:
    model, stats, feature_columns = load_artifacts()
    has_target = "selling_price" in df.columns
    proc_df = transform(df, stats)
    X = proc_df.drop(columns=["selling_price"]) if has_target else proc_df
    X = X.reindex(columns=feature_columns, fill_value=0)
    return pd.Series(model.predict(X), index=df.index, name="predicted_selling_price")


if __name__ == "__main__":
    sample = pd.read_csv("../cars24-car-price-cleaned-new.csv").sample(5, random_state=1)
    preds = predict(sample)
    result = sample[["make", "model", "selling_price"]].copy()
    result["predicted_selling_price"] = preds
    print(result.to_string())
