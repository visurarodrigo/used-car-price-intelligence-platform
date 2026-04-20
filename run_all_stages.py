from __future__ import annotations

import subprocess
import sys
import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

# Run the notebooks in dependency order so each stage can consume prior outputs.
NOTEBOOKS = [
    PROJECT_ROOT / "01-eda" / "stage1_eda.ipynb",
    PROJECT_ROOT / "02-baseline-modeling" / "stage2_baseline_modeling.ipynb",
    PROJECT_ROOT / "03-model-refinement" / "stage3_model_refinement.ipynb",
]

# Stage 04 and Stage 05 are script-based outputs generated after the notebooks.
STAGE4_SCRIPT = PROJECT_ROOT / "04-ensemble-modeling" / "stage4_ensemble_modeling.py"
STAGE5_SCRIPT = PROJECT_ROOT / "05-explainability" / "stage5_explainability.py"
STAGE7_SCRIPT = PROJECT_ROOT / "07-data-validation" / "stage7_data_validation.py"
STAGE8_SCRIPT = PROJECT_ROOT / "08-productionization" / "stage8_productionization.py"
STAGE6_SCRIPT = PROJECT_ROOT / "06-inference-api" / "stage6_inference_api.py"


def run_notebook(notebook_path: Path) -> None:
    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")

    # Execute the notebook in place so the saved outputs stay with the source file.
    print(f"\n[RUNNING] {notebook_path.relative_to(PROJECT_ROOT)}")
    cmd = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--inplace",
        str(notebook_path),
    ]
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
    print(f"[DONE]    {notebook_path.relative_to(PROJECT_ROOT)}")


def run_script(script_path: Path) -> None:
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    # Run standalone stage scripts with the project root as the working directory.
    print(f"\n[RUNNING] {script_path.relative_to(PROJECT_ROOT)}")
    cmd = [sys.executable, str(script_path)]
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
    print(f"[DONE]    {script_path.relative_to(PROJECT_ROOT)}")


def validate_inference_api(script_path: Path) -> None:
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    print(f"\n[RUNNING] {script_path.relative_to(PROJECT_ROOT)} (validation)")

    spec = importlib.util.spec_from_file_location("stage6_inference_api", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for: {script_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    module._load_runtime_artifacts()
    health = module.health()

    print(f"[DONE]    {script_path.relative_to(PROJECT_ROOT)} (validation)")
    print(f"[INFO]    Stage 6 model source: {health.get('model_source', 'unknown')}")
    print(f"[INFO]    Stage 6 feature count: {health.get('feature_count', 'unknown')}")


def run_stage7_validation(script_path: Path) -> None:
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    print(f"\n[RUNNING] {script_path.relative_to(PROJECT_ROOT)}")
    cmd = [sys.executable, str(script_path)]
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
    print(f"[DONE]    {script_path.relative_to(PROJECT_ROOT)}")


def main() -> int:
    # One command executes the full project pipeline from raw data to explanations.
    print("Executing used-car project stages in sequence...")
    try:
        for notebook in NOTEBOOKS:
            run_notebook(notebook)
        run_script(STAGE4_SCRIPT)
        run_script(STAGE5_SCRIPT)
        run_stage7_validation(STAGE7_SCRIPT)
        run_script(STAGE8_SCRIPT)
        validate_inference_api(STAGE6_SCRIPT)
    except subprocess.CalledProcessError as exc:
        print(f"\nExecution failed with exit code {exc.returncode}.")
        return exc.returncode
    except FileNotFoundError as exc:
        print(f"\n{exc}")
        return 1

    print("\nAll stages completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
