"""
models/train_model.py
=====================
Trains a Random Forest regression model that predicts a post's engagement
rate (%) from its features, then persists the fitted pipeline and evaluation
metrics to disk.

The whole preprocessing + model chain is wrapped in a single scikit-learn
``Pipeline`` (with a ``ColumnTransformer`` for encoding/scaling). Saving the
pipeline means inference code only has to hand raw feature values to
``pipeline.predict`` — no manual encoding required.

Run from the project root:
    python models/train_model.py
"""

from __future__ import annotations

import json
import os
import sys

# Allow running this file directly (python models/train_model.py) by putting
# the project root on the import path.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config
from utils.data_loader import load_data


def build_pipeline() -> Pipeline:
    """Construct the preprocessing + model pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                config.CATEGORICAL_FEATURES,
            ),
            ("num", StandardScaler(), config.NUMERIC_FEATURES),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=250,
        max_depth=18,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def feature_importances(pipeline: Pipeline) -> list[dict]:
    """Extract readable feature importances from the fitted pipeline."""
    pre = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]

    # Recover expanded feature names (one-hot columns + numeric names).
    cat_names = list(
        pre.named_transformers_["cat"].get_feature_names_out(config.CATEGORICAL_FEATURES)
    )
    names = cat_names + config.NUMERIC_FEATURES
    importances = model.feature_importances_

    paired = sorted(zip(names, importances), key=lambda x: x[1], reverse=True)
    return [
        {"feature": name, "importance": round(float(imp), 4)}
        for name, imp in paired[:15]
    ]


def train() -> dict:
    """Train, evaluate, and persist the model. Returns the metrics dict."""
    print("Loading data...")
    df = load_data()

    features = config.CATEGORICAL_FEATURES + config.NUMERIC_FEATURES
    X = df[features]
    y = df[config.TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training on {len(X_train):,} rows, testing on {len(X_test):,} rows...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # Evaluation ------------------------------------------------------------
    preds = pipeline.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))

    metrics = {
        "model": "RandomForestRegressor",
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4),
        "target_mean": round(float(y.mean()), 4),
        "feature_importances": feature_importances(pipeline),
    }

    # Persist ---------------------------------------------------------------
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, config.MODEL_PATH)
    with open(config.METRICS_PATH, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    print("\n=== Training complete ===")
    print(f"Model  -> {config.MODEL_PATH}")
    print(f"Metrics-> {config.METRICS_PATH}")
    print(f"R2={metrics['r2']}  RMSE={metrics['rmse']}  MAE={metrics['mae']}")
    return metrics


if __name__ == "__main__":
    train()
