# Used Car Price Intelligence Platform - Project Report

## 1. Project Summary

This project implements a professional, end-to-end machine learning platform for used car price prediction. The workflow is structured as a rigorous 8-stage pipeline, transitioning from raw data exploration to a production-ready inference service with integrated MLOps monitoring.

The platform is organized into eight connected stages:
1. **Stage 01 - EDA & Data Cleaning**: Quality assessment and creation of a canonical cleaned dataset.
2. **Stage 02 - Baseline Modeling**: Multi-model benchmarking to establish a performance floor.
3. **Stage 03 - Model Refinement**: Advanced tuning using polynomial features and Ridge regularization.
4. **Stage 04 - Ensemble Modeling**: Implementation of blending and stacking for robust prediction.
5. **Stage 05 - Explainability**: Feature influence analysis via permutation importance.
6. **Stage 06 - Inference API**: A FastAPI service with a sophisticated model-loading strategy.
7. **Stage 07 - Automated Data Validation**: Schema and range checks to ensure input quality.
8. **Stage 08 - Productionization**: Health checks, drift monitoring, and automated retraining logic.

The primary goal is to provide accurate, explainable valuations while ensuring the system remains stable in a production environment through continuous monitoring and lifecycle management.

## 2. Dataset Overview

The platform utilizes `data/usedcars.csv` as the primary data source.

**Key Statistics:**
- **Raw Records**: 500
- **Raw Features**: 29
- **Target Column**: `price`
- **Cleaned Dataset**: `01-eda/outputs/processed/usedcars_stage1.csv`
- **Processing**: The cleaned dataset expands to **69 columns** after one-hot encoding categorical variables and applying median imputation for missing numeric values. This canonical dataset serves as the single source of truth for all downstream modeling stages.

## 3. Stage-by-Stage Results

### Stage 01 - EDA and Cleaning
Focused on understanding data distributions and establishing a clean handoff.
- **Findings**: `price` exhibits a right-skewed distribution; strong predictive signals were identified in both numerical (engine size, horsepower) and categorical (body style, drive wheels) features.
- **Outcome**: Exported a fully numeric, zero-missing-value dataset to ensure consistent results across all subsequent stages.
- **Preview**: [Stage 01 Output Previews](01-eda/OUTPUT_PREVIEWS.md)

### Stage 02 - Baseline Modeling
Established a performance baseline by comparing multiple regression families.
- **Method**: Benchmarked Linear, Ridge, Lasso, ElasticNet, Polynomial, Random Forest, and Gradient Boosting models using 5-fold cross-validation.
- **MLOps**: Integrated **MLflow** for experiment tracking, logging every model's hyperparameters and metrics.
- **Best Baseline**: Gradient Boosting (Test R²: 0.9605, RMSE: 1841.79).
- **Preview**: [Stage 02 Output Previews](02-baseline-modeling/OUTPUT_PREVIEWS.md)

### Stage 03 - Model Refinement
Improved generalization through regularization and systematic tuning.
- **Method**: Combined polynomial feature expansion (degree 2) with Ridge regression, utilizing **Grid Search** to optimize the `alpha` penalty.
- **Diagnostics**: Implemented residual analysis to check for bias and heteroscedasticity, ensuring the model meets standard regression assumptions.
- **Preview**: [Stage 03 Output Previews](03-model-refinement/OUTPUT_PREVIEWS.md)

### Stage 04 - Ensemble Modeling
Increased robustness by combining the strongest learners.
- **Method**: Evaluated weighted blending (GBR + RF) and a Stacking Regressor with a Ridge meta-learner.
- **Outcome**: The Gradient Boosting model remained the strongest single learner, but ensemble strategies provided a more stable prediction baseline.
- **Improvement**: Reduced RMSE by approximately **174.70** compared to the Stage 02 baseline.
- **Preview**: [Stage 04 Output Previews](04-ensemble-modeling/OUTPUT_PREVIEWS.md)

### Stage 05 - Explainability
Translated the "black box" model into actionable business insights.
- **Method**: Used permutation importance to quantify how much each feature contributes to the model's accuracy.
- **Top Predictors**: `engine-size`, `curb-weight`, `horsepower`, and `highway-mpg` were identified as the primary drivers of car price.
- **Preview**: [Stage 05 Output Previews](05-explainability/OUTPUT_PREVIEWS.md)

### Stage 06 - Inference API
Deployed the model as a production-grade FastAPI service.
- **Advanced Loading Strategy**:
  1. **MLflow Model Registry**: Attempts to load the "Champion" model tagged as `Production`.
  2. **Local Artifacts**: Falls back to the Stage 08 deployed model or Stage 05 explainable model.
  3. **Emergency Retraining**: Can automatically retrain a compatible model from the cleaned dataset if all other loads fail, ensuring 100% service availability.
- **Key Endpoints**: Includes `/predict`, `/predict-batch`, and `/validate` (integrating Stage 07).
- **Documentation**: Integrated Swagger UI for easy API exploration.

### Stage 07 - Automated Data Validation
Implemented a "guardrail" layer to prevent model degradation.
- **Method**: Derived a validation profile (means, std, ranges) from the training data.
- **Logic**: The API uses this profile to check incoming payloads for missing fields, unknown features, and out-of-range values before prediction.
- **Outcome**: Ensures "garbage in, garbage out" is prevented at the API gateway.
- **Preview**: [Stage 07 Output Previews](07-data-validation/OUTPUT_PREVIEWS.md)

### Stage 08 - Productionization
Operationalized the model with a closed-loop monitoring and retraining system.
- **Health Check**: Continuously compares current performance against the Stage 05 RMSE baseline.
- **Drift Detection**: Implements feature-shift alerts using z-scores to detect when production data diverges from training data.
- **Automated Retraining**: Triggers a full model retrain on the latest available data if performance drops below the threshold or significant drift is detected.
- **Preview**: [Stage 08 Output Previews](08-productionization/OUTPUT_PREVIEWS.md)

## 4. Final Result Summary

| Stage | Model / Strategy | Test R² | Test RMSE | Test MAE |
|---|---|---:|---:|---:|
| **Stage 02** | Baseline (Gradient Boosting) | 0.9605 | 1841.79 | 1082.22 |
| **Stage 04** | Ensemble (Gradient Boosting) | 0.9676 | 1667.09 | 923.55 |
| **Stage 05** | Explainable GBR | 0.9676 | 1667.09 | 923.55 |
| **Stage 08** | Production Deployed Model | 0.9676 | 1667.09 | 923.55 |

**Overall Impact**: The pipeline reduced the average prediction error (RMSE) by ~175 units while adding a complete layer of explainability and production safety.

## 5. MLOps & Monitoring Architecture

The platform transcends a simple notebook project by implementing a full MLOps lifecycle:

- **Experiment Tracking (MLflow)**: Every model iteration is logged with its specific hyperparameters and metrics, eliminating ad-hoc note-taking and ensuring 100% reproducibility.
- **Model Registry**: The "Champion" model is versioned and promoted to `Production` stage within the MLflow Registry, allowing the API to update models without code changes.
- **Data Drift Monitoring (Evidently AI)**: A dedicated monitoring suite runs statistical tests (e.g., Kolmogorov-Smirnov) to detect feature drift. An interactive HTML report flags specifically which features have shifted, providing a leading indicator of model decay.
- **The Synergy**: 
  `Evidently (Drift)` $\rightarrow$ `Stage 08 (Retraining Trigger)` $\rightarrow$ `MLflow (Versioning)` $\rightarrow$ `Inference API (Consumption)`.

## 6. Practical Takeaway

This project demonstrates a production-ready approach to machine learning. By combining high-performance tree-based models with a rigorous validation layer, an automated monitoring system, and a sophisticated API deployment strategy, the platform ensures that car price predictions are not only accurate but also **stable, explainable, and maintainable** in a real-world environment.

## Author
Visura Rodrigo
