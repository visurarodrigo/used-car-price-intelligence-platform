# Stage 03: Model Refinement and Advanced Tuning

This folder contains a comprehensive model refinement workflow that improves upon baseline predictions through hyperparameter optimization and rigorous validation. This is **Stage 3** of the **Used Car Price Intelligence Platform** project.

## Purpose

Transform baseline models into production-grade predictive solutions by:
- Reducing overfitting and improving generalization
- Systematic hyperparameter tuning with Grid Search
- Multi-fold cross-validation for robust stability assessment
- In-depth error analysis and residual diagnostics
- Polynomial feature engineering with Ridge regularization

## What This Stage Covers

- **Data Preprocessing**: Automated feature scaling and encoding from `../data/usedcars.csv`
- **Train/Test Split**: Standard 80/20 split with stratified sampling
- **Baseline Reference Model**: Linear regression for performance comparison
- **Cross-Validation**: 5-fold CV for stability and consistency checking
- **Feature Engineering**: Polynomial expansion (degree 2) combined with Ridge L2 regularization
- **Hyperparameter Optimization**: Grid Search to systematically tune Ridge `alpha` parameter
- **Advanced Validation**: R², MSE, MAE, and residual analysis across folds
- **Diagnostic Visualization**: Prediction scatter plots and residual error patterns

## Output Structure

All generated outputs are automatically saved to:

- `outputs/figures/` - Diagnostic and evaluation charts
  - `predicted_vs_actual.png` - Actual vs. predicted scatter plot with perfect-fit diagonal
  - `residual_diagnostics.png` - Residual analysis (scatter + distribution)

## Files

- `Used Car Price Prediction.ipynb` - Main refinement and tuning notebook (clean, output-free source)
- `OUTPUT_PREVIEWS.md` - Visual gallery with interpretation of all exported diagnostics

## Dataset Context

- **Source**: `../data/usedcars.csv`
- **Rows**: 500 observations 
- **Prediction Target**: Price (continuous)
- **Note**: Inherits feature engineering and preprocessing logic from Stages 1 and 2

## Key Improvements Over Stage 2

- Feature expansion via polynomial transformation improves fit on non-linear patterns
- Ridge regularization addresses multicollinearity and overfitting risk
- Systematic Grid Search replaces manual tuning
- Cross-validation provides more honest generalization estimates
- Enhanced diagnostics reveal bias and heteroscedasticity patterns

## How to Run

1. Install dependencies:
   `pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter`
2. Open and run all cells:
   `Used Car Price Prediction.ipynb`
3. Check generated outputs for diagnostics:
   - Review `Predicted vs Actual Car Prices.png` for fit quality
   - Review `Residual Plot.png` for error patterns
   - See `OUTPUT_PREVIEWS.md` for interpretation

## Notes

- The notebook is intentionally committed without execution outputs for clean version control.
- All visual exports are generated dynamically at runtime.
- Residuals should be approximately normally distributed with constant variance for reliable predictions.
- Compare `alpha` tuning results in cross-validation to assess regularization trade-offs.

## Author

Visura Rodrigo




