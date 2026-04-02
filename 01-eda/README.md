# Stage 01: Exploratory Data Analysis (EDA)

This folder contains a fully refreshed and professional EDA workflow for the Used Car Price Intelligence Platform.

## Objectives

- Assess data quality (missing values, duplicates, dtypes, and edge cases)
- Understand target variable (`price`) behavior
- Analyze numerical and categorical feature distributions
- Evaluate feature relationships with `price` for downstream modeling
- Export publication-ready charts to a structured output directory
- **Perform full data cleaning and prepare canonical dataset for downstream stages**

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

## Data Cleaning Pipeline

After all EDA analysis, Stage 1 performs full data preparation:

1. **Categorical encoding**: One-hot encodes all non-numeric features (drop_first=True)
2. **Target validation**: Removes rows with missing `price` values
3. **Numeric conversion**: Converts all columns to numeric type
4. **Imputation**: Fills missing numeric values with column median
5. **Export**: Saves fully cleaned dataset to `outputs/processed/usedcars_stage1.csv`

**Output Dataset**:
- Rows: 500 (after dropping rows with missing target)
- Columns: 69 (expanded from 29 due to one-hot encoding)
- Missing values: 0
- All values: numeric (float64 or int64)
- Ready for immediate use in downstream stages without further preprocessing

## How to Run

1. Install dependencies:
   `pip install pandas numpy matplotlib seaborn jupyter`
2. Open and run notebook from project root:
   `01-eda/stage1_eda.ipynb`
3. Run all cells to:
   - Generate fresh chart files under `01-eda/outputs/`
   - Export cleaned dataset to `01-eda/outputs/processed/usedcars_stage1.csv`

## Notes

- All visual exports are saved programmatically via a reusable save helper.
- The notebook is intentionally committed without execution outputs for clean version control.
- Quick chart preview: see `OUTPUT_PREVIEWS.md`.
- The cleaned dataset export at the end ensures all downstream stages use consistent, preprocessed data.
- One-hot encoding converts 13 categorical features into 40 binary features (drop_first=True avoids multicollinearity).

