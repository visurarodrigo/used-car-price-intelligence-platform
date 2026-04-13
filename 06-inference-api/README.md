# Stage 06: Inference API

This stage exposes the trained model as a lightweight API for live price prediction.

## What It Uses

- Model artifact: `05-explainability/outputs/models/stage5_explainable_model.joblib`
- Feature schema source: `01-eda/outputs/processed/usedcars_stage1.csv`

The API expects Stage 1 cleaned feature names (numeric and one-hot encoded columns).

## Endpoints

- `GET /`
  - Simple landing response with links to docs and health checks
- `GET /health`
  - Basic service status and loaded feature count
- `GET /features`
  - Returns the exact feature list expected by the model
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
- Unknown features are ignored and returned in response as `unknown_features_ignored`.
- If you retrain with new features, restart the API so it reloads the new schema/model.
- If loading the saved model fails due environment compatibility, the API automatically retrains a compatible Gradient Boosting model from the Stage 1 cleaned dataset.
- If you added the root endpoint recently, stop and restart Uvicorn so the browser picks up the new code.
