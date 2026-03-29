# Used Car Price Intelligence Platform

End-to-end machine learning project for used car price prediction using exploratory analysis, baseline regression, and refined model optimization.

## Project Overview

This repository presents one unified project that moves through three connected stages:
- `01-eda`: exploratory data analysis and visual insights
- `02-linear-polynomial`: baseline linear vs polynomial regression modeling
- `03-model-refinement`: model improvement with validation and regularization

Dataset summary:
- 201 records
- 29 features
- Target: `price`
- Shared dataset path: `data/usedcars.csv`

## Business Problem

Used car pricing is difficult due to:
- Non-linear relationships between vehicle features and price
- Risk of underpricing (lost margin)
- Risk of overpricing (slower sales)

This project builds a data-driven pricing approach to support more consistent and explainable valuation.

## Key Features / What This Project Does

- Performs structured EDA to identify major price drivers
- Compares baseline regression approaches
- Builds improved models with polynomial features and Ridge regularization
- Uses cross-validation and error metrics for robust evaluation
- Organizes work into clear stage-based notebooks for reproducibility

## Project Structure

```text
used-car-price-intelligence-platform/
|-- README.md
|-- data/
|   `-- usedcars.csv
|-- 01-eda/
|   |-- EDA - UsedCars.ipynb
|   |-- boxplots-categorical/
|   `-- regression-plots/
|-- 02-linear-polynomial/
|   `-- Used Cars Price Prediction.ipynb
`-- 03-model-refinement/
    `-- Used Car Price Prediction.ipynb
```

## Methods & Models Used

- Data analysis: descriptive statistics, correlation analysis, distribution and category plots
- Feature engineering: one-hot encoding, polynomial feature expansion, scaling
- Models:
  - Linear Regression
  - Polynomial Regression
  - Ridge Regression
- Validation:
  - Train/test split
  - 4-fold cross-validation
  - Grid search for Ridge `alpha`
- Metrics: R-squared, Mean Squared Error (MSE), residual analysis

## Results / Insights

Model comparison highlights:
- Linear Regression (single feature): R2 = 0.761, MSE = 15,021,126
- Polynomial Regression (single feature): R2 = 0.761, MSE = 15,016,547
- Polynomial Pipeline (3 features): R2 = 0.828, MSE = 10,819,563

Key takeaways:
- Non-linear modeling improved fit over a basic linear baseline
- Multi-feature modeling significantly increased predictive performance
- Regularization and validation improved model reliability

## How to Run

1. Clone the repository
```bash
git clone https://github.com/yourusername/used-car-price-intelligence-platform.git
cd used-car-price-intelligence-platform
```

2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

3. Launch notebooks
```bash
jupyter notebook
```

4. Run in sequence:
- `01-eda/EDA - UsedCars.ipynb`
- `02-linear-polynomial/Used Cars Price Prediction.ipynb`
- `03-model-refinement/Used Car Price Prediction.ipynb`

## Future Improvements

- Add ensemble models (Random Forest, XGBoost)
- Expand dataset size and feature coverage
- Expose model through an API or lightweight dashboard
- Add model versioning and automated retraining pipeline

## Author

Visura Rodrigo
