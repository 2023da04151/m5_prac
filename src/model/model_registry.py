"""Register an MLflow model version from experiment_info.json.

Run from the project root:
    python src/model/model_registry.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import mlflow
from mlflow import MlflowClient

PROJECT_DIR = Path.cwd()
DEFAULT_ENV_PATH = PROJECT_DIR / ".env"
DEFAULT_EXPERIMENT_INFO_PATH = PROJECT_DIR / "experiment_info.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("model_registry")


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


def load_experiment_info(file_path: Path) -> dict[str, Any]:
    """Load MLflow run information saved by model_evaluation.py."""
    if not file_path.exists():
        raise FileNotFoundError(f"Experiment info file not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as file:
        experiment_info = json.load(file)

    required_keys = {"run_id", "model_path"}
    missing_keys = required_keys - set(experiment_info)
    if missing_keys:
        raise ValueError(f"{file_path} is missing keys: {sorted(missing_keys)}")

    return experiment_info


def register_model(
    experiment_info: dict[str, Any],
    model_name: str,
    alias: str,
    validation_status: str,
) -> None:
    """Register a model from an MLflow run and set alias/tag metadata."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise ValueError("MLFLOW_TRACKING_URI is missing from .env")
    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()
    model_uri = f"runs:/{experiment_info['run_id']}/{experiment_info['model_path']}"
    logger.info("Registering model from %s as %s", model_uri, model_name)

    result = mlflow.register_model(model_uri=model_uri, name=model_name)
    version = str(result.version)
    logger.info("Registered %s version %s", result.name, version)

    client.set_registered_model_alias(result.name, alias, version)
    logger.info("Set alias %s on %s version %s", alias, result.name, version)

    client.set_model_version_tag(
        result.name,
        version,
        "validation_status",
        validation_status,
    )
    logger.info("Set validation_status=%s on %s version %s", validation_status, result.name, version)


def parse_args() -> argparse.Namespace:
    """Parse model registry options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-info", type=Path, default=DEFAULT_EXPERIMENT_INFO_PATH)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_PATH)
    parser.add_argument("--model-name", default="m5_forecasting_model")
    parser.add_argument("--alias", default="champion")
    parser.add_argument("--validation-status", default="staging")
    return parser.parse_args()


def main() -> None:
    """Script entry point."""
    args = parse_args()
    load_env_file(args.env_file)
    experiment_info = load_experiment_info(args.experiment_info)
    register_model(
        experiment_info,
        args.model_name,
        args.alias,
        args.validation_status,
    )


if __name__ == "__main__":
    main()
