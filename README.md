# Customer Retention Risk Application

A locally runnable machine-learning application that estimates customer churn risk. It combines a pre-trained scikit-learn model, a FastAPI server, automated tests, and a browser interface built with HTML, CSS, and JavaScript.

The project uses deterministic synthetic data. It contains no real customer records and must not be used for real customer or business decisions.

## Application flow

```text
Browser form
    ↓ JSON request
FastAPI /predict endpoint
    ↓ validated feature row
Preprocessing + classification pipeline
    ↓ probability and risk level
Browser result panel
```

## API endpoints

| Endpoint | Purpose |
| --- | --- |
| `/` | Browser interface |
| `/predict` | Accepts a customer profile and returns churn risk |
| `/health` | Confirms the server and model artifact can load |
| `/metadata` | Returns application and model metadata |
| `/docs` | Interactive OpenAPI documentation |

## Data source

The dataset is generated in `scripts/train_model.py` with NumPy using a fixed random seed of 42. The default dataset contains 2,400 fictional customer profiles.

| Feature | Generation rule |
| --- | --- |
| `tenure_months` | Integer from 0 through 72 |
| `monthly_charges` | Normal distribution centered at 72, clipped to 18–140 |
| `contract` | Month-to-month, one-year, or two-year |
| `internet_service` | DSL, fiber optic, or none |
| `tech_support` | Yes or no |
| `payment_method` | Electronic check, mailed check, bank transfer, or credit card |

The binary target represents whether a fictional customer churned. Its probability is generated from an invented relationship in which month-to-month contracts, fiber service, no technical support, electronic checks, and higher charges increase synthetic risk. Longer tenure and a two-year contract reduce it. Random noise is added before the score is converted to a probability.

These relationships are application examples, not findings about real customers. There is no raw external dataset or ETL feed. The generation script is the data source and transformation record.

## Data preparation and model

The training pipeline performs two transformations:

- `StandardScaler` standardizes the two numeric features.
- `OneHotEncoder(handle_unknown="ignore")` converts categorical values into model inputs.

`ColumnTransformer` combines those transformations, and a scikit-learn `Pipeline` stores preprocessing with the classifier. This ensures that prediction uses the same transformations used during training.

Logistic Regression is the default estimator because it is small, fast, produces probabilities, and provides a clear classification baseline. An optional Random Forest is also available.

The current metadata reports training accuracy. Because the original example fits and evaluates on the same synthetic rows, that number is not an unbiased estimate of generalization. A real project should use separate validation and test data and consider precision, recall, F1, ROC-AUC, calibration, class balance, business costs, temporal validation, subgroup performance, and fairness.

## Model artifacts

```text
models/churn_model.joblib   serialized preprocessing and estimator
models/metadata.json        model type, versions, metric, source, and purpose
```

Only load joblib artifacts from trusted sources. Python deserialization can execute code, and scikit-learn artifacts can be sensitive to framework versions. The preflight script verifies the installed version against the metadata.

## Requirements

- Python 3.12 recommended; Python 3.11 or newer required
- Git
- Docker Desktop for the environment readiness check
- macOS or Windows PowerShell

See `prerequisite.pdf` before beginning setup.

## Local setup

### macOS

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python scripts/preflight.py --full
```

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python scripts/preflight.py --full
```

If PowerShell blocks activation, run this in the current PowerShell window and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

## Run the tests

```bash
python -m pytest -q
```

The tests cover server health, page loading, prediction responses, request validation, model output, metadata, and preflight behavior.

## Start the server

```bash
python -m uvicorn app.main:app --reload --port 8000
```

Open:

- http://localhost:8000
- http://localhost:8000/docs
- http://localhost:8000/health
- http://localhost:8000/metadata

Stop the server with `Ctrl+C`.

## Example profiles

A higher-risk fictional profile:

```json
{
  "tenure_months": 4,
  "monthly_charges": 98.0,
  "contract": "Month-to-month",
  "internet_service": "Fiber optic",
  "tech_support": "No",
  "payment_method": "Electronic check"
}
```

A lower-risk fictional profile:

```json
{
  "tenure_months": 60,
  "monthly_charges": 55.0,
  "contract": "Two year",
  "internet_service": "DSL",
  "tech_support": "Yes",
  "payment_method": "Credit card"
}
```

## Regenerate or change the model

Recreate the default Logistic Regression artifact:

```bash
python scripts/train_model.py --algorithm logistic
python -m pytest -q
```

Create the included Random Forest alternative:

```bash
python scripts/train_model.py --algorithm random_forest
python -m pytest -q
```

Both commands replace `models/churn_model.joblib` and `models/metadata.json`. Review and commit the two files together. To use real data, replace `build_dataset()` with a governed loading and transformation process that produces the same feature columns and a binary target. Add schema validation, leakage checks, train/validation/test splitting, data-quality reporting, privacy controls, and appropriate evaluation before considering real use.

## Project structure

```text
app/                    FastAPI server, model loading, schemas, and frontend
models/                 Serialized pipeline and model metadata
scripts/preflight.py    Cross-platform environment and artifact check
scripts/train_model.py  Synthetic data generation and model training
tests/                  API, model, and preflight tests
requirements.txt        Runtime dependencies
requirements-dev.txt    Test and development dependencies
prerequisite.pdf        Participant readiness checklist
```

## Limitations

- All data and target relationships are synthetic.
- The reported training accuracy does not measure real-world generalization.
- Risk thresholds are illustrative application settings.
- The application has no authentication, authorization, audit trail, monitoring, drift detection, experiment tracking, model registry, or automated retraining.
- Predictions must not be used for consequential customer decisions.

## License

MIT License. Copyright 2026 Ogochukwu Ozotta.
