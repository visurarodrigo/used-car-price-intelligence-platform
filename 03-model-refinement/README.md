# Stage 03: Model Refinement and Advanced Tuning

This folder contains a comprehensive model refinement workflow that improves upon baseline predictions through hyperparameter optimization and rigorous validation. This is **Stage 3** of the **Used Car Price Intelligence Platform** project.

## Purpose

Transform baseline models into production-grade predictive solutions by:
- Reducing overfitting and improving generalization
- Systematic hyperparameter tuning with Grid Search
- Multi-fold cross-validation for robust stability assessment
- In-depth error analysis and residual diagnostics
- Polynomial feature engineering with Ridge regularization

**Key Design Pattern**: Uses the same pre-cleaned Stage 1 dataset as Stage 2, enabling direct performance comparison while focusing on advanced regularization and tuning techniques.

## What This Stage Covers

- **Data Loading**: Reads pre-cleaned dataset from Stage 01 (`01-eda/outputs/processed/usedcars_stage1.csv`)
  - All numeric, no missing values, no categorical encoding needed
- **Preprocessing**: StandardScaler applied within cross-validation pipeline (prevents data leakage)
- **Train/Test Split**: Standard 80/20 split
- **Baseline Reference Model**: Linear regression for performance comparison
- **Cross-Validation**: 5-fold CV for stability and consistency checking
- **Feature Engineering**: Polynomial expansion (degree 2) combined with Ridge L2 regularization
- **Hyperparameter Optimization**: Grid Search to systematically tune Ridge `alpha` parameter across [0.001, 0.01, 0.1, 1, 10, 100, 1000]
- **Advanced Validation**: R², MSE, MAE, and residual analysis across folds
- **Diagnostic Visualization**: Prediction scatter plots and residual error patterns

## Output Structure

All generated outputs are automatically saved to:

- `outputs/figures/` - Diagnostic and evaluation charts
  - `predicted_vs_actual.png` - Actual vs. predicted scatter plot with perfect-fit diagonal
  - `residual_diagnostics.png` - Residual analysis (scatter + distribution)

## Files

- `stage3_model_refinement.ipynb` - Main refinement and tuning notebook (clean, output-free source)
- `OUTPUT_PREVIEWS.md` - Visual gallery with interpretation of all exported diagnostics

## Dataset Context

- **Source**: Stage 01 cleaned snapshot (`01-eda/outputs/processed/usedcars_stage1.csv`)
- **Rows**: 500 observations 
- **Columns**: 69 numeric features (expanded via one-hot encoding in Stage 1)
- **Missing Values**: 0 (all handled in Stage 1)
- **Prediction Target**: Price (continuous)
- **Data Quality**: All numeric, ready for immediate modeling
- **Baseline Link**: Loads Stage 2 saved metrics for direct performance comparison

## Key Improvements Over Stage 2

- **Polynomial Features**: Expansion improves fit on non-linear patterns
- **Ridge Regularization**: Addresses multicollinearity and overfitting via systematic L2 penalty
- **Systematic Tuning**: Grid Search over ridge alpha instead of fixed parameter
- **Honest CV Loop**: Scaling inside cross-validation (prevents test data leakage)
- **Enhanced Diagnostics**: Detailed residual analysis reveals bias and heteroscedasticity
- **Focused Refinement**: Leverages pre-cleaned Stage 1 data, no preprocessing overhead

## How to Run

1. Install dependencies:
   `pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter`
2. Open and run all cells:
   `stage3_model_refinement.ipynb`
3. Check generated outputs for diagnostics:
   - Review `Predicted vs Actual Car Prices.png` for fit quality
   - Review `Residual Plot.png` for error patterns
   - See `OUTPUT_PREVIEWS.md` for interpretation

## Notes

- The notebook is intentionally committed without execution outputs for clean version control.
- All visual exports are generated dynamically at runtime.
- Data is pre-cleaned from Stage 1; focus is on model refinement, not data wrangling.
- Residuals should be approximately normally distributed with constant variance for reliable predictions.
- Compare `alpha` tuning results in cross-validation to assess regularization trade-offs.
- StandardScaler is applied inside the cross-validation pipeline to prevent information leakage from test set into training loop.

## Author

Visura Rodrigo




