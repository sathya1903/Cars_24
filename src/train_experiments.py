"""Compare candidate algorithms on a held-out test split.

Run this to decide which algorithm + hyperparameters to lock in for
train_final_model.py. Not used in production - experimentation only.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from preprocessing import TARGET, fit_preprocessing_stats, load_data, split_features_target, transform

DATA_PATH = "../cars24-car-price-cleaned-new.csv"
RANDOM_STATE = 42


def evaluate(name, y_true, y_pred, elapsed):
    print(
        f"{name:<28} R2={r2_score(y_true, y_pred):.4f}  "
        f"MAE={mean_absolute_error(y_true, y_pred):.4f}  "
        f"RMSE={root_mean_squared_error(y_true, y_pred):.4f}  "
        f"({elapsed:.1f}s)"
    )


def main():
    df = load_data(DATA_PATH)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)

    stats = fit_preprocessing_stats(train_df)
    train_proc = transform(train_df, stats)
    test_proc = transform(test_df, stats)

    X_train, y_train = split_features_target(train_proc)
    X_test, y_test = split_features_target(test_proc)

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}\n")

    print("=" * 90)
    print("Baseline models (default hyperparameters)")
    print("=" * 90)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    t0 = time.time()
    ridge = Ridge(random_state=RANDOM_STATE).fit(X_train_scaled, y_train)
    evaluate("Ridge (scaled)", y_test, ridge.predict(X_test_scaled), time.time() - t0)

    t0 = time.time()
    rf = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1).fit(X_train, y_train)
    evaluate("RandomForest (default)", y_test, rf.predict(X_test), time.time() - t0)

    t0 = time.time()
    gbr = GradientBoostingRegressor(random_state=RANDOM_STATE).fit(X_train, y_train)
    evaluate("GradientBoosting (default)", y_test, gbr.predict(X_test), time.time() - t0)

    t0 = time.time()
    xgb = XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1).fit(X_train, y_train)
    evaluate("XGBoost (default)", y_test, xgb.predict(X_test), time.time() - t0)

    print()
    print("=" * 90)
    print("Hyperparameter search for the two best candidates (RandomForest, XGBoost)")
    print("=" * 90)

    rf_grid = GridSearchCV(
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        param_grid={
            "n_estimators": [300, 500],
            "max_depth": [None, 15, 25],
            "min_samples_leaf": [1, 2],
        },
        scoring="r2",
        cv=3,
        n_jobs=-1,
    )
    t0 = time.time()
    rf_grid.fit(X_train, y_train)
    evaluate(f"RandomForest (tuned) {rf_grid.best_params_}", y_test, rf_grid.best_estimator_.predict(X_test), time.time() - t0)

    xgb_grid = GridSearchCV(
        XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        param_grid={
            "n_estimators": [300, 500],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.03, 0.05, 0.1],
        },
        scoring="r2",
        cv=3,
        n_jobs=-1,
    )
    t0 = time.time()
    xgb_grid.fit(X_train, y_train)
    evaluate(f"XGBoost (tuned) {xgb_grid.best_params_}", y_test, xgb_grid.best_estimator_.predict(X_test), time.time() - t0)

    print()
    print("Best RandomForest params:", rf_grid.best_params_)
    print("Best XGBoost params:", xgb_grid.best_params_)
    print(
        "\nUse the winning algorithm + params above to update FINAL_MODEL "
        "in train_final_model.py."
    )


if __name__ == "__main__":
    main()
