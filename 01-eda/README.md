# Stage 01: Exploratory Data Analysis (EDA)

This folder is **Stage 1** of the full **Used Car Price Intelligence Platform** project.

## Purpose

Build a strong understanding of the data before modeling:
- Check data quality and missing values
- Explore feature distributions
- Analyze relationships between vehicle attributes and price
- Identify early signals for feature selection

## What This Stage Covers

- Univariate analysis for numeric and categorical variables
- Correlation and relationship exploration
- Price behavior across key categories (body style, drive wheels, fuel system)
- Visual diagnostics through histograms, regression plots, and box plots

## Files

- `EDA - UsedCars.ipynb` - Main notebook for data understanding
- `regression-plots/` - Relationship plots for key numeric features
- `boxplots-categorical/` - Price distributions across categorical groups
- Additional exported figures used for reporting

## Dataset

- Shared dataset path: `../data/usedcars.csv`
- Records: 201
- Features: 29
- Target: `price`

## How to Run

1. Install dependencies:
	`pip install pandas numpy matplotlib seaborn jupyter`
2. Open and run:
	`EDA - UsedCars.ipynb`

## Author

Visura Rodrigo

