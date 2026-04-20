# Used Car Price Intelligence Platform - Project Report

## 1. Project Summary

This project predicts used car prices with a full machine learning workflow. It starts with data exploration, moves through data cleaning and model building, and ends with ensemble modeling, explainability, data validation, and productionization.

The project is organized in eight stages:

1. Stage 01 - Exploratory data analysis and cleaning
2. Stage 02 - Baseline modeling
3. Stage 03 - Model refinement
4. Stage 04 - Ensemble modeling
5. Stage 05 - Explainability
6. Stage 06 - Inference API
7. Stage 07 - Automated data validation
8. Stage 08 - Productionization (monitoring, retraining, and lightweight deployment)

The goal is simple: build a model that predicts car prices accurately and explain why the model makes those predictions.

## 2. Dataset Overview

The project uses `data/usedcars.csv` as the raw input data.

Key facts:

- Raw records: 500
- Raw features: 29
- Target column: `price`
- Cleaned dataset for downstream stages: `01-eda/outputs/processed/usedcars_stage1.csv`
- Cleaned dataset format: numeric-only after one-hot encoding and median imputation

## 3. Stage-by-Stage Results

### Stage 01 - EDA and Cleaning

This stage explored the data, checked quality issues, and exported a clean dataset for modeling.

Main results:

- Missing values were limited and manageable
- `price` was right-skewed
- Several numeric and categorical features showed useful relationships with the target
- A cleaned modeling dataset was exported for all later stages

Output preview:

- [Stage 01 Output Previews](01-eda/OUTPUT_PREVIEWS.md)

Important artifact:

- [Cleaned dataset](01-eda/outputs/processed/usedcars_stage1.csv)

### Stage 02 - Baseline Modeling

This stage compared several regression models and selected the best baseline.

Best model:

- Gradient Boosting

Metrics:

- Test R2: 0.96048233541031
- Test RMSE: 1841.788643661087
- Test MAE: 1082.2233602468232

Output preview:

- [Stage 02 Output Previews](02-baseline-modeling/OUTPUT_PREVIEWS.md)

Key artifacts:

- [Model comparison CSV](02-baseline-modeling/outputs/metrics/model_comparison.csv)
- [Best model metrics](02-baseline-modeling/outputs/metrics/best_model_metrics.json)
- [Saved best model](02-baseline-modeling/outputs/models/best_model.joblib)

### Stage 03 - Model Refinement

This stage tested polynomial features, Ridge regression, and grid search tuning.

Main results:

- The refined model was checked with train/test evaluation and cross-validation
- Diagnostic plots were created to inspect prediction quality and residual behavior
- This stage focused on stability and error patterns rather than a new production model

Output preview:

- [Stage 03 Output Previews](03-model-refinement/OUTPUT_PREVIEWS.md)

Key artifacts:

- [Prediction plot](03-model-refinement/outputs/figures/predicted_vs_actual.png)
- [Residual diagnostics](03-model-refinement/outputs/figures/residual_diagnostics.png)

### Stage 04 - Ensemble Modeling

This stage combined strong tree models and tested ensemble strategies.

Best model:

- Gradient Boosting

Metrics:

- Test R2: 0.9676234321930117
- Test RMSE: 1667.0919691154925
- Test MAE: 923.5545603562045
- RMSE improvement vs Stage 02: -174.69667454559453

Best blend weight:

- Gradient Boosting weight: 0.9

Output preview:

- [Stage 04 Output Previews](04-ensemble-modeling/OUTPUT_PREVIEWS.md)

Key artifacts:

- [Ensemble comparison CSV](04-ensemble-modeling/outputs/metrics/ensemble_comparison.csv)
- [Best ensemble metrics](04-ensemble-modeling/outputs/metrics/best_ensemble_metrics.json)
- [Saved best ensemble model](04-ensemble-modeling/outputs/models/best_ensemble_model.joblib)
- [Ensemble prediction plot](04-ensemble-modeling/outputs/figures/ensemble_predicted_vs_actual.png)

### Stage 05 - Explainability

This stage retrained the winning model in the current environment and explained the predictions.

Model:

- Gradient Boosting Regressor

Metrics:

- Test R2: 0.9676234321930117
- Test RMSE: 1667.0919691154925
- Test MAE: 923.5545603562045
- RMSE delta vs Stage 04: 0.0

Top features from permutation importance:

- engine-size
- curb-weight
- horsepower
- highway-mpg
- drive-wheels_rwd
- width
- bore
- length
- make_bmw
- peak-rpm

Output preview:

- [Stage 05 Output Previews](05-explainability/OUTPUT_PREVIEWS.md)

Key artifacts:

- [Feature importance CSV](05-explainability/outputs/metrics/feature_importance.csv)
- [Stage 05 metrics](05-explainability/outputs/metrics/stage5_model_metrics.json)
- [Feature importance plot](05-explainability/outputs/figures/feature_importance.png)
- [Prediction scatter plot](05-explainability/outputs/figures/prediction_scatter.png)
- [Saved explainable model](05-explainability/outputs/models/stage5_explainable_model.joblib)

### Stage 06 - Inference API

This stage wraps the final trained model as a lightweight HTTP API for real-time inference.

Main results:

- Added health, feature schema, single prediction, and batch prediction endpoints
- Added a root landing route so the base URL returns a helpful JSON message instead of 404
- Loads Stage 08 deployed model artifact when available, with Stage 05 fallback
- Aligns incoming request data to the Stage 1 cleaned feature schema
- Returns unknown input fields in the response for easier debugging

Key artifacts:

- [Stage 06 API code](06-inference-api/stage6_inference_api.py)
- [Stage 06 README](06-inference-api/README.md)
- [Sample payload](06-inference-api/sample_predict_payload.json)

### Stage 07 - Automated Data Validation

This stage adds validation checks for incoming data before it is passed into the modeling pipeline.

Main results:

- Schema validation for required fields
- Type and range checks for numeric inputs
- Missing-value thresholds and rejection rules
- Unknown-field detection and payload sparsity warnings
- Validation profile and sample report artifacts for API integration

Key artifacts:

- [Stage 07 script](07-data-validation/stage7_data_validation.py)
- [Validation profile](07-data-validation/outputs/metrics/data_validation_profile.json)
- [Sample validation report](07-data-validation/outputs/metrics/sample_validation_report.json)

### Stage 08 - Productionization

This stage operationalizes the model with monitoring, retraining logic, and deployment-ready assets.

Main results:

- Added RMSE baseline monitoring against Stage 05 metrics
- Added lightweight feature-shift monitoring between reference and current evaluation data
- Added retraining decision logic with threshold-based triggers
- Published a deployment-ready model artifact for the inference API
- Generated lightweight deployment files (Dockerfile, docker-compose, and run scripts)

Monitoring snapshot:

- Current Test R2: 0.9676234321930117
- Current Test RMSE: 1667.0919691154925
- Current Test MAE: 923.5545603562045
- RMSE delta vs Stage 05 baseline: 0.0
- Retraining triggered: No

Key artifacts:

- [Stage 08 script](08-productionization/stage8_productionization.py)
- [Monitoring report](08-productionization/outputs/metrics/monitoring_report.json)
- [Deployment manifest](08-productionization/outputs/metrics/deployment_manifest.json)
- [Deployed model](08-productionization/outputs/models/deployed_model.joblib)

## 4. Final Result Summary

| Stage | Main Result | Test R2 | Test RMSE | Test MAE |
|---|---|---:|---:|---:|
| Stage 02 | Best baseline: Gradient Boosting | 0.96048233541031 | 1841.788643661087 | 1082.2233602468232 |
| Stage 04 | Best ensemble: Gradient Boosting | 0.9676234321930117 | 1667.0919691154925 | 923.5545603562045 |
| Stage 05 | Explainable Gradient Boosting Regressor | 0.9676234321930117 | 1667.0919691154925 | 923.5545603562045 |
| Stage 06 | Inference API for live prediction | N/A | N/A | N/A |
| Stage 07 | Automated payload validation before inference | N/A | N/A | N/A |
| Stage 08 | Monitoring + deployment-ready model (no retraining needed) | 0.9676234321930117 | 1667.0919691154925 | 923.5545603562045 |

Overall improvement:

- Stage 04 and Stage 05 reduced RMSE by about 174.70 compared to the Stage 02 baseline.
- The final model kept strong accuracy while adding explainability.

## 5. Output Preview Index

This project includes a preview document for each stage:

- [Stage 01 Preview Gallery](01-eda/OUTPUT_PREVIEWS.md)
- [Stage 02 Preview Gallery](02-baseline-modeling/OUTPUT_PREVIEWS.md)
- [Stage 03 Preview Gallery](03-model-refinement/OUTPUT_PREVIEWS.md)
- [Stage 04 Preview Gallery](04-ensemble-modeling/OUTPUT_PREVIEWS.md)
- [Stage 05 Preview Gallery](05-explainability/OUTPUT_PREVIEWS.md)

These files are the best place to quickly review charts, metrics files, and model artifacts for each stage.

## 6. How the Project Flows

1. Raw data is loaded from `data/usedcars.csv`.
2. Stage 01 cleans and exports a reusable modeling dataset.
3. Stage 02 benchmarks several regression models.
4. Stage 03 refines the modeling approach and checks residual behavior.
5. Stage 04 builds ensemble models and picks the best one.
6. Stage 05 explains the final model with permutation importance.
7. Stage 06 serves the trained model through API endpoints for live prediction.
8. Stage 07 validates new input data before inference.
9. Stage 08 monitors production quality, retrains when needed, and prepares deployment assets.

You can run the full pipeline with:

```bash
python run_all_stages.py
```

## 7. Practical Meaning of the Results

The project shows that the used car price problem has strong predictive signal. The cleaned data and tree-based models were enough to reach strong performance, and the ensemble step improved the error score further.

The explainability stage adds value because it shows which features matter most. That makes the project easier to understand because you can explain both the model quality and the business meaning of the model.

## 8. Simple Takeaway

The simple message is:

- I built an end-to-end machine learning pipeline for used car price prediction.
- I improved the model from a strong baseline to a better ensemble.
- I finished the project with validation, monitoring/retraining logic, and a lightweight deployment layer.

## Author

Visura Rodrigo