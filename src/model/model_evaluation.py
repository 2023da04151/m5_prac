"""Evaluate saved LightGBM predictions and log results to MLflow.

Expected files:
    models/non_recur_CA_1_cv4_predictions.csv
    models/model.pkl

Run from the project root:
    python src/model/model_evaluation.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
import yaml

PROJECT_DIR = Path.cwd()
DEFAULT_PARAMS_PATH = PROJECT_DIR / "params.yaml"
DEFAULT_ENV_PATH = PROJECT_DIR / ".env"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("model_evaluation")


def load_env_file(env_path: Path) -> None:
    """Load KEY=value pairs from a local .env file into environment variables."""
    if not env_path.exists():
        logger.warning("%s not found. Using existing environment variables.", env_path)
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip().strip('"').strip("'")


def load_params(params_path: Path) -> dict[str, Any]:
    """Load model-evaluation parameters from params.yaml."""
    with params_path.open("r", encoding="utf-8") as file:
        params = yaml.safe_load(file) or {}

    evaluation_params = params.get("model_evaluation")
    if not isinstance(evaluation_params, dict):
        raise TypeError(f"{params_path} must contain a 'model_evaluation' section.")
    return evaluation_params


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate regression metrics for forecast evaluation."""
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    error = y_pred - y_true
    total = float(np.abs(y_true).sum())
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "wape": float(np.abs(error).sum() / total) if total > 0 else float("nan"),
    }


def evaluate_predictions(predictions_path: Path) -> dict[str, float]:
    """Read validation predictions and calculate metrics."""
    predictions = pd.read_csv(predictions_path)
    required_columns = {"y_true", "y_pred"}
    missing_columns = required_columns - set(predictions.columns)
    if missing_columns:
        raise ValueError(f"{predictions_path} is missing columns: {sorted(missing_columns)}")

    return regression_metrics(
        predictions["y_true"].to_numpy(),
        predictions["y_pred"].to_numpy(),
    )



def log_to_mlflow(
    metrics: dict[str, float],
    model_path: Path,
    experiment_name: str,
    run_name: str,
) -> dict[str, str]:
    """Log evaluation metrics and a registerable LightGBM model to MLflow."""

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    username = os.getenv("MLFLOW_TRACKING_USERNAME")
    password = os.getenv("MLFLOW_TRACKING_PASSWORD")

    if not tracking_uri:
        raise ValueError("MLFLOW_TRACKING_URI is missing from .env")
    if not username or not password:
        raise ValueError("MLflow username or password is missing from .env")

    mlflow.set_tracking_uri(tracking_uri)

    # Configure Basic Authentication for the MLflow server.
    os.environ["MLFLOW_TRACKING_USERNAME"] = username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = password

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_metrics(metrics)
        with model_path.open("rb") as file:
            model = pickle.load(file)
        if not isinstance(model, lgb.Booster):
            raise TypeError(f"Expected {model_path} to contain a lightgbm.Booster.")

        mlflow.lightgbm.log_model(lgb_model=model, artifact_path="model")
        mlflow.log_artifact(str(model_path), artifact_path="raw_model")
        logger.info("Logged metrics and LightGBM model to MLflow run %s", run.info.run_id)

        return {
            "run_id": run.info.run_id,
            "model_path": "model",
        }



def save_experiment_info(experiment_info: dict[str, str], output_path: Path) -> None:
    """Save MLflow run details for later pipeline stages."""
    output_path.write_text(json.dumps(experiment_info, indent=2), encoding="utf-8")
    logger.info("Saved experiment info to %s", output_path)


def parse_args() -> argparse.Namespace:
    """Parse optional command-line overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS_PATH)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_PATH)
    return parser.parse_args()


def main() -> None:
    """Script entry point."""
    args = parse_args()
    load_env_file(args.env_file)
    params = load_params(args.params)

    predictions_path = Path(params["predictions_path"])
    model_path = Path(params["model_path"])
    experiment_name = params.get("experiment_name", "m5_forecasting_experiment")
    run_name = params.get("run_name", "model_evaluation")
    experiment_info_path = Path(params.get("experiment_info_path", "experiment_info.json"))

    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Pickle model file not found: {model_path}")

    metrics = evaluate_predictions(predictions_path)
    logger.info("Evaluation metrics: %s", {key: round(value, 4) for key, value in metrics.items()})
    experiment_info = log_to_mlflow(metrics, model_path, experiment_name, run_name)
    save_experiment_info(experiment_info, experiment_info_path)


if __name__ == "__main__":
    main()
