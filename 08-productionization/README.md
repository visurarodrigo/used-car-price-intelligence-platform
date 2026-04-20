# Stage 08: Productionization (Monitoring, Retraining, and Deployment)

This stage closes the project by operationalizing the final model.

## What It Does

- Runs lightweight model monitoring against a held-out evaluation split
- Compares current RMSE against the Stage 05 baseline threshold
- Computes feature mean-shift alerts to catch potential data drift
- Applies retraining logic automatically when alert conditions are met
- Publishes a deployment-ready model artifact for the inference API
- Generates lightweight deployment files (Dockerfile, docker-compose, local run scripts)

## Outputs

- `outputs/metrics/monitoring_report.json`
- `outputs/metrics/deployment_manifest.json`
- `outputs/models/deployed_model.joblib`
- `outputs/deployment/Dockerfile`
- `outputs/deployment/docker-compose.yml`
- `outputs/deployment/start_api.ps1`
- `outputs/deployment/start_api.sh`

## Run

From project root:

```bash
python 08-productionization/stage8_productionization.py
```

## Deployment Notes

- Stage 06 API automatically prefers `08-productionization/outputs/models/deployed_model.joblib` when present.
- If Stage 08 is not run yet, Stage 06 falls back to the Stage 05 model artifact.
- Docker deployment from project root:

```bash
docker build -f 08-productionization/outputs/deployment/Dockerfile -t used-car-price-intelligence:latest .
docker run -p 8000:8000 used-car-price-intelligence:latest
```
