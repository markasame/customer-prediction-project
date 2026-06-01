#!/usr/bin/env python3
"""
run_pipeline.py — Run the full churn-prediction training pipeline.

Executes three sequential steps:
  1. Data ingestion   (downloads Telco CSV, validates schema)
  2. Feature engineering (preprocessing, SMOTE, saves pipeline artifact)
  3. Model training   (XGBoost, evaluation, saves model artifact)

Usage:
    python run_pipeline.py                  # full pipeline
    python run_pipeline.py --skip-ingest    # skip download (data already present)
    python run_pipeline.py --skip-features  # skip feature step (processed CSV exists)
    python run_pipeline.py --serve          # train then start API + dashboard
"""
import argparse
import os
import subprocess
import sys
import time

MIN_PYTHON = (3, 11)

PIPELINE_STEPS = [
    ("Data ingestion",      "src/data/ingest.py"),
    ("Feature engineering", "src/features/build.py"),
    ("Model training",      "src/models/train.py"),
]

ROOT = os.path.dirname(os.path.abspath(__file__))


def banner(text: str) -> None:
    bar = "=" * 62
    print(f"\n{bar}\n  {text}\n{bar}")


def check_python() -> None:
    if sys.version_info < MIN_PYTHON:
        sys.exit(
            f"ERROR: Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ required "
            f"(running {sys.version.split()[0]})"
        )


def run_step(label: str, script: str) -> None:
    banner(label)
    result = subprocess.run(
        [sys.executable, script],
        cwd=ROOT,
    )
    if result.returncode != 0:
        print(f"\n[FAILED] {label} exited with code {result.returncode}")
        sys.exit(result.returncode)
    print(f"\n[OK] {label}")


def serve() -> None:
    print("\nStarting services ...")
    print("  API:       http://localhost:8000  (Swagger: /docs)")
    print("  Dashboard: http://localhost:8501")
    print("\n  Press Ctrl+C to stop both.\n")

    api_proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "src.api.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
        ],
        cwd=ROOT,
    )
    time.sleep(3)

    try:
        subprocess.run(
            [
                sys.executable, "-m", "streamlit", "run",
                "dashboard/app.py",
                "--server.port", "8501",
                "--server.address", "0.0.0.0",
            ],
            cwd=ROOT,
        )
    except KeyboardInterrupt:
        pass
    finally:
        api_proc.terminate()
        api_proc.wait()
        print("\nServices stopped.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the churn-prediction pipeline end-to-end."
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip data download (data/raw/telco.csv already exists)",
    )
    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="Skip feature engineering (data/processed/features.csv already exists)",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start API + dashboard after training completes",
    )
    args = parser.parse_args()

    check_python()

    banner(f"Churn Prediction Pipeline  |  Python {sys.version.split()[0]}")

    skip = set()
    if args.skip_ingest:
        skip.add("src/data/ingest.py")
    if args.skip_features:
        skip.add("src/features/build.py")

    t0 = time.time()
    for label, script in PIPELINE_STEPS:
        if script in skip:
            print(f"\n[SKIP] {label}")
            continue
        run_step(label, script)

    elapsed = time.time() - t0
    banner(f"Pipeline complete in {elapsed:.0f}s")
    print("  Model artifact : models/churn_model.pkl")
    print("  Pipeline artifact: models/pipeline.pkl")
    print("  Processed data : data/processed/features.csv")

    if args.serve:
        serve()
    else:
        print("\nNext steps:")
        print("  Start API:        uvicorn src.api.main:app --reload")
        print("  Start dashboard:  streamlit run dashboard/app.py")
        print("  Run tests:        pytest tests/")
        print("  Run with serve:   python run_pipeline.py --serve")


if __name__ == "__main__":
    main()
