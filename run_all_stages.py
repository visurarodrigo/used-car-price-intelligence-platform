from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

NOTEBOOKS = [
    PROJECT_ROOT / "01-eda" / "stage1_eda.ipynb",
    PROJECT_ROOT / "02-baseline-modeling" / "stage2_baseline_modeling.ipynb",
    PROJECT_ROOT / "03-model-refinement" / "stage3_model_refinement.ipynb",
]


def run_notebook(notebook_path: Path) -> None:
    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")

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


def main() -> int:
    print("Executing used-car project notebooks in sequence...")
    try:
        for notebook in NOTEBOOKS:
            run_notebook(notebook)
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
