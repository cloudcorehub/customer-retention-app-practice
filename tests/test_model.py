from app.model import load_metadata, predict_churn
from app.schemas import CustomerFeatures


def test_model_metadata_is_present():
    metadata = load_metadata()
    assert metadata["model_version"] != "unknown"
    assert metadata["model_type"] != "unknown"


def test_prediction_is_valid_probability_and_risk():
    sample = CustomerFeatures(
        tenure_months=4,
        monthly_charges=98.0,
        contract="Month-to-month",
        internet_service="Fiber optic",
        tech_support="No",
        payment_method="Electronic check",
    )
    probability, risk = predict_churn(sample)
    assert 0.0 <= probability <= 1.0
    assert risk in {"Low", "Medium", "High"}
