# MLflow + Evidently AI — Full Implementation Guide
## Used Car Price Intelligence Platform

---

## Phase 1: MLflow Setup & The Tracking UI

### What we will do
Install MLflow via pip and start the MLflow Tracking Server locally on your Windows machine. Verify the UI loads correctly in your browser before touching any code.

### The Concept
MLflow runs a local web server (usually on `localhost:5000`). When you open it in your browser, you get a dashboard that replaces the chaos of ad-hoc experiment notes. Instead of digging through folders to find which model had which hyperparameters, you see a clean table comparing every experiment run — side by side, sortable by any metric.

### Step-by-step

**1.1 — Install MLflow**
```bash
pip install mlflow
```
Verify: `mlflow --version` should print a version number.

**1.2 — Create a dedicated MLflow tracking folder**
In your project root, create a folder called `mlruns/` — MLflow will write all experiment data here automatically. Add it to `.gitignore` (we will handle what to commit in Phase 5).

**1.3 — Start the tracking server**
```bash
mlflow ui
```
Open `http://localhost:5000` in your browser. You should see the MLflow UI with an empty "Default" experiment. Keep this terminal open while you work — the UI updates live as runs complete.

**1.4 — Create a named experiment**
Instead of using "Default", create one named experiment for this project:
```python
import mlflow
mlflow.set_experiment("used-car-price-intelligence")
```
This line goes at the top of each training script you instrument.

---

## Phase 2: Instrumenting Your Training Scripts

### What we will do
Add MLflow tracking hooks to your existing Stage 2, 3, 4, and 5 training scripts. Log parameters, metrics, and model artifacts for every model that trains.

### The Concept
"Instrumentation" means wrapping your existing training code inside an `mlflow.start_run()` context. MLflow then intercepts everything you tell it to log. Three things get logged per run:
- **Parameters** — the knobs you set before training (e.g., `n_estimators=500`, `learning_rate=0.05`)
- **Metrics** — the scores produced after training (RMSE, MAE, R²)
- **Artifacts** — the actual trained model files (`.joblib`) and any plots

### Step-by-step

**2.1 — Basic run structure**
Wrap every model's training block like this:
```python
import mlflow
import mlflow.sklearn

mlflow.set_experiment("used-car-price-intelligence")

with mlflow.start_run(run_name="XGBoost Baseline"):
    # --- your existing training code here ---
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Log parameters
    mlflow.log_param("n_estimators", 500)
    mlflow.log_param("learning_rate", 0.05)
    mlflow.log_param("max_depth", 6)

    # Log metrics
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("R2", r2)

    # Log model artifact
    mlflow.sklearn.log_model(model, artifact_path="model")
```

**2.2 — Use autolog where possible**
For Scikit-learn and XGBoost, MLflow can log everything automatically:
```python
mlflow.sklearn.autolog()   # for scikit-learn models
mlflow.xgboost.autolog()   # for XGBoost
```
Place this once at the top of the script, before `.fit()` is called. It captures params, metrics, and model files with no extra lines.

**2.3 — Scripts to instrument**
Apply to all of these — give each run a clear `run_name`:
- Stage 2: baseline models (Linear Regression, Ridge, Lasso, Decision Tree)
- Stage 3: intermediate models (Random Forest, Gradient Boosting, XGBoost)
- Stage 4: ensemble blending
- Stage 5: ensemble stacking

After running all scripts, your MLflow UI should show 10+ runs under the `used-car-price-intelligence` experiment.

**2.4 — Log diagnostic plots as artifacts**
If you have SHAP plots or residual plots, log them too:
```python
fig.savefig("residuals.png")
mlflow.log_artifact("residuals.png")
```

---

## Phase 3: The MLflow Model Registry

### What we will do
Use MLflow's Model Registry to register your best model (likely the stacking ensemble from Stage 5) with a version tag. Promote it to "Champion" stage.

### The Concept
The Model Registry is version control for ML models — like Git, but for trained artifacts. You register a model, give it a name, and assign it a lifecycle stage: `Staging`, `Production`, or `Archived`. This proves to interviewers that you understand model lifecycle management, not just model training.

### Step-by-step

**3.1 — Register the best model from a run**
In the MLflow UI, find your best run (lowest RMSE / highest R²). Click the run → click the model artifact → click "Register Model". Name it: `used-car-price-champion`.

Or do it in code after the run:
```python
run_id = "paste-your-best-run-id-here"
model_uri = f"runs:/{run_id}/model"
mlflow.register_model(model_uri, "used-car-price-champion")
```

**3.2 — Promote to Production stage**
```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
client.transition_model_version_stage(
    name="used-car-price-champion",
    version=1,
    stage="Production"
)
```

**3.3 — Load the Production model in inference**
Update your FastAPI inference script to load from the registry instead of a local path:
```python
model = mlflow.pyfunc.load_model("models:/used-car-price-champion/Production")
```
This is how real ML teams serve models — the API doesn't care which file path, it loads whatever is tagged Production.

**3.4 — Screenshot the registry**
Open `http://localhost:5000/#/models` in your browser. Take a screenshot showing your model name, version number, and "Production" stage tag — this goes in your README.

---

## Phase 4: Evidently AI & Data Drift Monitoring

### What we will do
Install Evidently AI. Write a script that compares your original training data against a simulated production dataset (with intentional drift injected). Generate an interactive HTML drift report.

### The Concept
In production, the real world changes. Car prices shift, new models enter the market, fuel prices spike. When your input data starts looking different from what your model was trained on, predictions degrade silently. Evidently detects this by running statistical tests (KS test, chi-square, Jensen-Shannon divergence) on each feature and flagging which ones have "drifted" — so you know when to retrigger retraining.

### Step-by-step

**4.1 — Install Evidently**
```bash
pip install evidently
```

**4.2 — Create the drift monitoring script**
Create a new file: `monitoring/drift_report.py`

```python
import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset

# Load your original training data
reference_data = pd.read_csv("data/processed/train.csv")

# Simulate production drift — alter 3 features to mimic real-world shift
production_data = reference_data.copy()
np.random.seed(42)

# Inject drift: engine-size increases (newer, larger cars in production)
production_data["engine-size"] = production_data["engine-size"] * np.random.uniform(1.05, 1.25, len(production_data))

# Inject drift: curb-weight shifts
production_data["curb-weight"] = production_data["curb-weight"] + np.random.normal(150, 50, len(production_data))

# Inject drift: horsepower distribution shifts
production_data["horsepower"] = production_data["horsepower"] * np.random.uniform(0.9, 1.3, len(production_data))

# Build and run the report
report = Report(metrics=[
    DataDriftPreset(),
    DataQualityPreset(),
])

report.run(reference_data=reference_data, current_data=production_data)
report.save_html("monitoring/drift_report.html")
print("Drift report saved to monitoring/drift_report.html")
```

**4.3 — Run the script**
```bash
python monitoring/drift_report.py
```
Open `monitoring/drift_report.html` in your browser. You will see:
- An overall drift summary (how many features drifted)
- Per-feature distribution plots showing reference vs production
- Statistical test results and p-values for each feature

**4.4 — Screenshot the report**
Take a screenshot of the overall drift summary and one feature's distribution comparison. Both go in your README.

---

## Phase 5: GitHub & Portfolio Polish

### What we will do
Update `.gitignore`, commit the right files, push to GitHub, and update your README with a new MLOps & Monitoring section including screenshots.

### The Concept
A portfolio project is only as strong as its presentation. A recruiter who opens your repo and sees a "MLOps & Monitoring" section with real screenshots of MLflow and Evidently will immediately know you think in production terms — not just notebook terms.

### Step-by-step

**5.1 — Update .gitignore**
Add these lines — MLflow stores large binary model files locally that should not go to GitHub:
```
# MLflow
mlruns/
mlartifacts/

# Evidently (keep the HTML report, ignore any large data exports)
monitoring/*.json
```
The `monitoring/drift_report.html` should NOT be in `.gitignore` — you want to commit and display this.

**5.2 — What to commit**
```
monitoring/
    drift_report.py          ← the script
    drift_report.html        ← the generated report
    screenshots/
        mlflow_ui.png        ← MLflow experiment runs table
        mlflow_registry.png  ← model registry showing Production stage
        evidently_summary.png ← drift summary page
        evidently_feature.png ← one feature's distribution comparison
```

**5.3 — Commit and push**
```bash
git add monitoring/
git add .gitignore
git commit -m "feat: add MLflow experiment tracking, model registry, and Evidently drift monitoring"
git push
```

**5.4 — Update README**
Add a new section called `## MLOps & Monitoring` after your existing Tech Stack section:

```markdown
## MLOps & Monitoring

### Experiment Tracking — MLflow
All model training runs are tracked with MLflow, logging hyperparameters,
metrics (RMSE, MAE, R²), and model artifacts for every experiment.
The best model (stacking ensemble, R² = 0.97) is registered in the
MLflow Model Registry and promoted to **Production** stage.

![MLflow UI](monitoring/screenshots/mlflow_ui.png)
![MLflow Registry](monitoring/screenshots/mlflow_registry.png)

### Data Drift Monitoring — Evidently AI
A drift monitoring script compares the training distribution against
a simulated production dataset. Evidently runs statistical tests
(KS test, Jensen-Shannon divergence) per feature and generates an
interactive HTML report flagging which features have drifted and
when retraining should be triggered.

![Evidently Drift Summary](monitoring/screenshots/evidently_summary.png)

📄 [View full drift report](monitoring/drift_report.html)
```

**5.5 — Final check before moving on**
Verify these 4 things are visible on your GitHub repo page:
- [ ] `monitoring/` folder exists with `drift_report.py` and `drift_report.html`
- [ ] README shows MLOps & Monitoring section with embedded screenshots
- [ ] MLflow UI screenshot shows 10+ runs with metrics columns
- [ ] MLflow Registry screenshot shows model name + Production stage tag

Once all 4 are checked, Prompt 1 is complete.

---

## Summary — what you will have after Phase 5

| Addition | Where visible | Recruiter signal |
|---|---|---|
| MLflow experiment runs | GitHub README screenshot | "Tracks experiments professionally" |
| MLflow model registry | GitHub README screenshot | "Understands model lifecycle" |
| Evidently drift report | GitHub `monitoring/drift_report.html` | "Thinks about production monitoring" |
| Updated README section | GitHub repo front page | "Presents work clearly" |