# Stage 06: Inference API

This stage exposes the trained model as a lightweight API for live price prediction.

## Model Management & MLflow Integration

The API is designed to be the final consumption point of the ML lifecycle. It employs a sophisticated loading strategy to ensure high availability and version control:

1.  **MLflow Model Registry (Preferred)**: The API first attempts to load the "Champion" model directly from the MLflow Model Registry (`models:/used-car-price-champion/Production`). This allows for zero-downtime model updates via the Registry.
2.  **Local Artifacts (Fallback)**: If the registry is unavailable, it falls back to local `.joblib` files (Stage 08 deployed model or Stage 05 explainable model).
3.  **Automatic Retraining (Emergency Fallback)**: In case of environment/version incompatibility, the API can automatically retrain a compatible Gradient Boosting model from the Stage 1 cleaned dataset to ensure the service remains online.

**To manage models via MLflow:**
Run the following command from the project root to view and promote models to Production:
`mlflow ui --backend-store-uri sqlite:///mlflow.db`
Then open `http://127.0.0.1:5000` in your browser.

## What It Uses

- Model artifact precedence:
  1. MLflow Model Registry (`Production` stage)
  2. `08-productionization/outputs/models/deployed_model.joblib`
  3. `05-explainability/outputs/models/stage5_explainable_model.joblib` (fallback)
- Feature schema source: `01-eda/outputs/processed/usedcars_stage1.csv`

The API expects Stage 1 cleaned feature names (numeric and one-hot encoded columns).

## Endpoints

- `GET /`
  - Simple landing response with links to docs and health checks
- `GET /health`
  - Basic service status and loaded feature count
- `GET /features`
  - Returns the exact feature list expected by the model
- `POST /validate`
  - Validates a payload against the Stage 1 feature schema and data ranges
- `POST /predict`
  - Single-row prediction
- `POST /predict-batch`
  - Multi-row prediction

## Setup

Install API dependencies:

```bash
pip install fastapi uvicorn
```

## Run

From project root:

```bash
python -m uvicorn stage6_inference_api:app --reload --app-dir 06-inference-api
```

## Sample Request

Single prediction:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d @06-inference-api/sample_predict_payload.json
```

Open interactive docs:

- `http://127.0.0.1:8000/docs`

## Notes

- Missing expected features are filled with `0.0` by default.
- Validation warnings are returned in the prediction response.
- Unknown features are ignored by default, but they are still reported by the validation endpoint.
- If you retrain with new features, restart the API so it reloads the new schema/model.
- Running Stage 08 refreshes the preferred deployed model artifact used by this API.
- If loading the saved model fails due environment compatibility, the API automatically retrains a compatible Gradient Boosting model from the Stage 1 cleaned dataset.
- If you added the root endpoint recently, stop and restart Uvicorn so the browser picks up the new code.
