from fastapi.testclient import TestClient
from app.main import app
from app.config import APP_NAME

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_version"] != "unknown"


def test_home_page_loads():
    response = client.get("/")
    assert response.status_code == 200
    assert APP_NAME in response.text


def test_prediction_endpoint():
    response = client.post(
        "/predict",
        json={
            "tenure_months": 8,
            "monthly_charges": 89.5,
            "contract": "Month-to-month",
            "internet_service": "Fiber optic",
            "tech_support": "No",
            "payment_method": "Electronic check",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["risk_level"] in {"Low", "Medium", "High"}


def test_prediction_rejects_out_of_range_input():
    response = client.post(
        "/predict",
        json={
            "tenure_months": 120,
            "monthly_charges": 89.5,
            "contract": "Month-to-month",
            "internet_service": "Fiber optic",
            "tech_support": "No",
            "payment_method": "Electronic check",
        },
    )
    assert response.status_code == 422


def test_metadata_identifies_synthetic_model():
    response = client.get("/metadata")
    assert response.status_code == 200
    body = response.json()
    assert body["dataset"] == "synthetic customer data"
    assert body["purpose"] == "demonstration only"
    assert body["framework_version"] == "1.8.0"
