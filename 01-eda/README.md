# Stage 01: Exploratory Data Analysis (EDA)

This folder contains a fully refreshed and professional EDA workflow for the Used Car Price Intelligence Platform.

## Objectives

- Assess data quality (missing values, duplicates, dtypes, and edge cases)
- Understand target variable (`price`) behavior
- Analyze numerical and categorical feature distributions
- Evaluate feature relationships with `price` for downstream modeling
- Export publication-ready charts to a structured output directory

## Folder Structure

- `EDA - UsedCars.ipynb`: Main EDA notebook (clean, output-free source)
- `OUTPUT_PREVIEWS.md`: Preview gallery of all exported EDA charts with short interpretations
- `outputs/overview/`: Dataset-level quality and target overview charts
- `outputs/numerical/`: Numerical feature distribution visuals
- `outputs/categorical/`: Categorical distribution and boxplot visuals
- `outputs/bivariate/`: Correlation and regression relationship charts

## Dataset Snapshot

- Source: `../data/usedcars.csv`
- Rows: 500 
- Columns: 29
- Target: `price`

## Analysis Highlights

- Structured quality checks for nulls, uniqueness, and duplicates
- Price diagnostics through histogram and boxplot
- Full numerical profiling and outlier scan (IQR-based)
- High-impact categorical comparisons against price
- Correlation heatmap and top regression plots with `price`
- Actionable EDA summary and next-step recommendations

## How to Run

1. Install dependencies:
   `pip install pandas numpy matplotlib seaborn jupyter`
2. Open and run notebook from project root:
   `01-eda/EDA - UsedCars.ipynb`
3. Run all cells to generate fresh chart files under `01-eda/outputs/`

## Notes

- All visual exports are saved programmatically via a reusable save helper.
- The notebook is intentionally committed without execution outputs for clean version control.
- Quick chart preview: see `OUTPUT_PREVIEWS.md`.

