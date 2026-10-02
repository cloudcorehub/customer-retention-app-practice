from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.config import APP_NAME, APP_SUBTITLE, APP_VERSION
from app.model import load_metadata, load_model, predict_churn
from app.schemas import CustomerFeatures, PredictionResponse

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="API for serving a pre-trained customer churn model.",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    html = html.replace("{{APP_NAME}}", APP_NAME)
    html = html.replace("{{APP_SUBTITLE}}", APP_SUBTITLE)
    html = html.replace("{{APP_VERSION}}", APP_VERSION)
    return HTMLResponse(html)


@app.get("/health")
def health() -> dict:
    load_model()
    metadata = load_metadata()
    return {
        "status": "ok",
        "app_version": APP_VERSION,
        "model_version": metadata.get("model_version", "unknown"),
    }


@app.get("/metadata")
def metadata() -> dict:
    return {"app_version": APP_VERSION, **load_metadata()}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures) -> PredictionResponse:
    probability, risk = predict_churn(features)
    metadata = load_metadata()
    return PredictionResponse(
        churn_probability=probability,
        risk_level=risk,
        model_version=metadata.get("model_version", "unknown"),
        model_type=metadata.get("model_type", "unknown"),
        app_version=APP_VERSION,
    )
