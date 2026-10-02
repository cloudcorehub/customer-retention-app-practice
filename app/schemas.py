from typing import Literal
from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    tenure_months: int = Field(ge=0, le=72)
    monthly_charges: float = Field(ge=10, le=200)
    contract: Literal["Month-to-month", "One year", "Two year"]
    internet_service: Literal["DSL", "Fiber optic", "None"]
    tech_support: Literal["Yes", "No"]
    payment_method: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer",
        "Credit card",
    ]


class PredictionResponse(BaseModel):
    churn_probability: float
    risk_level: Literal["Low", "Medium", "High"]
    model_version: str
    model_type: str
    app_version: str
