"""Create the deterministic synthetic model artifact used by the application.

The pre-trained model is committed for reproducible local use. Run this script
when you intentionally want to regenerate or change the artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"


def build_dataset(rows: int = 2400, seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)

    tenure = rng.integers(0, 73, rows)
    monthly = np.clip(rng.normal(72, 27, rows), 18, 140).round(2)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], rows, p=[0.55, 0.25, 0.20])
    internet = rng.choice(["DSL", "Fiber optic", "None"], rows, p=[0.35, 0.50, 0.15])
    support = rng.choice(["Yes", "No"], rows, p=[0.42, 0.58])
    payment = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        rows,
        p=[0.34, 0.18, 0.24, 0.24],
    )

    # Synthetic churn relationship: month-to-month, fiber, higher charges and
    # no tech support increase risk; tenure and longer contracts reduce it.
    logit = (
        -0.7
        + 1.25 * (contract == "Month-to-month")
        - 0.75 * (contract == "Two year")
        + 0.75 * (internet == "Fiber optic")
        + 0.65 * (support == "No")
        + 0.55 * (payment == "Electronic check")
        + 0.018 * (monthly - 70)
        - 0.027 * tenure
        + rng.normal(0, 0.55, rows)
    )
    probability = 1 / (1 + np.exp(-logit))
    target = rng.binomial(1, probability)

    X = pd.DataFrame(
        {
            "tenure_months": tenure,
            "monthly_charges": monthly,
            "contract": contract,
            "internet_service": internet,
            "tech_support": support,
            "payment_method": payment,
        }
    )
    return X, target


def train(algorithm: str = "logistic") -> dict:
    X, y = build_dataset()
    numeric = ["tenure_months", "monthly_charges"]
    categorical = ["contract", "internet_service", "tech_support", "payment_method"]

    preprocessing = ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )

    if algorithm == "random_forest":
        estimator = RandomForestClassifier(n_estimators=180, max_depth=8, random_state=42)
        model_type = "Random Forest"
        model_version = "2.0.0"
    else:
        estimator = LogisticRegression(max_iter=600, random_state=42)
        model_type = "Logistic Regression"
        model_version = "1.0.0"

    pipeline = Pipeline([("preprocess", preprocessing), ("model", estimator)])
    pipeline.fit(X, y)
    predictions = pipeline.predict(X)
    accuracy = float(accuracy_score(y, predictions))

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_DIR / "churn_model.joblib")
    metadata = {
        "model_version": model_version,
        "model_type": model_type,
        "framework": "scikit-learn",
        "framework_version": sklearn.__version__,
        "training_accuracy": round(accuracy, 4),
        "dataset": "synthetic customer data",
        "purpose": "demonstration only",
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=["logistic", "random_forest"], default="logistic")
    args = parser.parse_args()
    print(json.dumps(train(args.algorithm), indent=2))
