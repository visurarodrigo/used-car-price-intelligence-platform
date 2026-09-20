# Stage 02: Professional Regression Benchmark

This folder is **Stage 2** of the **Used Car Price Intelligence Platform** project.

## Purpose

Build a regression benchmark pipeline that compares multiple model families, selects the best candidate for downstream stages, and establishes a performance baseline.

**Key Design Pattern**: This stage consumes pre-cleaned data from Stage 1, enabling focus on model comparison without data preprocessing complexity.

## What This Stage Covers

- **Data Loading**: Reads pre-cleaned dataset from Stage 01 (`01-eda/outputs/processed/usedcars_stage1.csv`)
  - All data is already numeric, no missing values
  - No imputation or encoding needed
- **Preprocessing**: 
  - Scaling for linear/polynomial models (StandardScaler in pipeline)
  - Tree models used as-is (no scaling required)
  - No categorical encoding (already done in Stage 1)
- **Train/Test Split**: 80/20 split with 5-fold cross-validation
- **Model Comparison**:
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - ElasticNet Regression
  - Polynomial Regression (degree 2)
  - Polynomial Regression (degree 3)
  - Random Forest Regressor
  - Gradient Boosting Regressor
- **Evaluation**: R², RMSE, and MAE (train + CV + test)
- **Automatic Best-Model Selection**: Based on test RMSE
- **Artifact Persistence**: Comparison table, best-model metrics JSON, serialized model joblib

## Experiment Tracking (MLflow)

This stage integrates **MLflow** to ensure every model benchmark is reproducible:
- **Automatic Logging**: Uses `mlflow.sklearn.autolog()` to capture hyperparameters and training metrics.
- **Run Structure**: Each candidate model is logged as a named run within the `used-car-price-intelligence` experiment.
- **Summary Run**: A final `Best_Model_Summary` run is created to store the winning model's artifacts and the global comparison table.

**To visualize the benchmarks:**
Run the following command from the project root:
`mlflow ui --backend-store-uri sqlite:///mlflow.db`
Then open `http://127.0.0.1:5000` in your browser.

## Output Structure

All generated outputs are saved automatically to:

- `outputs/figures/`
- `outputs/metrics/`
- `outputs/models/`

## Files

- `stage2_baseline_modeling.ipynb` - Main Stage 02 benchmark notebook

## How to Run

1. Install dependencies:
   `pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter`
2. Open and run all cells in:
   `stage2_baseline_modeling.ipynb`

## Author

Visura Rodrigo
