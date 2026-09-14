# Stage 08: Productionization (Monitoring, Retraining, and Deployment)

This stage closes the project by operationalizing the final model, ensuring it meets quality gates before being promoted to production.

## What It Does

- **Model Monitoring**: Runs a lightweight health check against a held-out evaluation split.
- **Performance Gating**: Compares current RMSE against the Stage 05 baseline. If degradation exceeds the alert ratio, retraining is triggered.
- **Drift Detection**: Computes feature mean-shift alerts (z-scores) to detect potential data drift between training and evaluation sets.
- **Automated Retraining**: If performance degrades or drift is detected, the model is automatically retrained on the full available dataset to ensure maximum stability.
- **Artifact Promotion**: Publishes the final, validated model artifact and a deployment manifest for the inference API.

## Outputs

- `outputs/metrics/monitoring_report.json`: Detailed results of the health check and the retraining decision log.
- `outputs/metrics/deployment_manifest.json`: Configuration file mapping the model artifact to the API entrypoint.
- `outputs/models/deployed_model.joblib`: The final, serialized model ready for production deployment.

## Run

From project root:

```bash
python 08-productionization/stage8_productionization.py
```

## Deployment Notes

- **API Integration**: The Stage 06 API automatically prefers `08-productionization/outputs/models/deployed_model.joblib` when present. If not yet generated, it falls back to the Stage 05 model.
- **Containerization**: Deployment is handled via the **root-level Dockerfile**.
- **Docker Build**:
  
  ```bash
  docker build -t used-car-price-intelligence:latest .
  docker run -p 8000:8000 used-car-price-intelligence:latest
  ```
