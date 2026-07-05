"""Train the finalized model on the full dataset and save artifacts.

Algorithm and hyperparameters below were selected by src/train_experiments.py,
which compared Ridge, RandomForest, GradientBoosting, and XGBoost via an
80/20 train_test_split plus grid search:

    XGBoost (tuned) n_estimators=500, max_depth=6, learning_rate=0.1
    R2=0.9572  MAE=0.6662  RMSE=1.0172   <- best of all candidates

This script re-trains that exact configuration on 100% of the data (no
held-out split) so the shipped model uses every available row, then saves
the model + preprocessing stats needed to serve predictions on new data.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
from xgboost import XGBRegressor

from preprocessing import fit_preprocessing_stats, load_data, split_features_target, transform

DATA_PATH = "../cars24-car-price-cleaned-new.csv"
ARTIFACTS_DIR = Path("../artifacts")
RANDOM_STATE = 42

FINAL_MODEL_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.1,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


def main():
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    df = load_data(DATA_PATH)
    stats = fit_preprocessing_stats(df)
    proc_df = transform(df, stats)
    X, y = split_features_target(proc_df)

    model = XGBRegressor(**FINAL_MODEL_PARAMS)
    model.fit(X, y)

    joblib.dump(model, ARTIFACTS_DIR / "model.joblib")
    joblib.dump(stats, ARTIFACTS_DIR / "preprocessing_stats.joblib")
    joblib.dump(list(X.columns), ARTIFACTS_DIR / "feature_columns.joblib")

    metadata = {
        "algorithm": "XGBRegressor",
        "hyperparameters": FINAL_MODEL_PARAMS,
        "n_training_rows": len(df),
        "n_features": X.shape[1],
        "target": "selling_price (INR lakhs)",
        "held_out_test_metrics": {
            "r2": 0.9572,
            "mae": 0.6662,
            "rmse": 1.0172,
            "note": "measured on 20% split in train_experiments.py before retraining on 100% of data",
        },
    }
    with open(ARTIFACTS_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Trained on {len(df)} rows, {X.shape[1]} features.")
    print(f"Artifacts saved to {ARTIFACTS_DIR.resolve()}")
    for p in sorted(ARTIFACTS_DIR.iterdir()):
        print(" -", p.name)


if __name__ == "__main__":
    main()
