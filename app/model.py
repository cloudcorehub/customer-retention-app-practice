from functools import lru_cache
import json
from pathlib import Path

import joblib
import pandas as pd

from app.config import HIGH_RISK_THRESHOLD, MEDIUM_RISK_THRESHOLD
from app.schemas import CustomerFeatures

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "churn_model.joblib"
METADATA_PATH = BASE_DIR / "models" / "metadata.json"


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model artifact not found at {MODEL_PATH}. Run: python scripts/train_model.py"
        )
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_metadata() -> dict:
    if not METADATA_PATH.exists():
        return {"model_version": "unknown", "model_type": "unknown"}
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def predict_churn(features: CustomerFeatures) -> tuple[float, str]:
    model = load_model()
    row = pd.DataFrame([features.model_dump()])
    probability = float(model.predict_proba(row)[0][1])

    if probability >= HIGH_RISK_THRESHOLD:
        risk = "High"
    elif probability >= MEDIUM_RISK_THRESHOLD:
        risk = "Medium"
    else:
        risk = "Low"

    return round(probability, 4), risk
