"""FastAPI app for predictions from the registered M5 LightGBM model.

Run from the project root with ``uvicorn m5_app:app --reload``. Submit only
``item_id``, ``store_id``, and ``day``; the app looks up the matching row in
``data/processed/prediction_features.csv`` and sends its trained features to
the registered model.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import lightgbm as lgb
import mlflow
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from mlflow import MlflowClient
from pydantic import BaseModel, Field

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_ENV_PATH = PROJECT_DIR / ".env"
REQUESTS_PATH = PROJECT_DIR / "prediction_requests.csv"
FEATURES_PATH = PROJECT_DIR / "data" / "processed" / "prediction_features.csv"
DEFAULT_META_PATH = PROJECT_DIR / "models" / "non_recur_CA_1_cv4_meta.json"
LOOKUP_COLUMNS = ["request_item_id", "request_store_id", "request_day"]
CATEGORICAL_FEATURES = [
    "item_id",
    "dept_id",
    "cat_id",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("m5_forecasting_api")


def load_env_file(env_path: Path) -> None:
    """Load local MLflow settings without overwriting existing environment."""
    if env_path.exists():
        load_dotenv(env_path, override=False)
    else:
        logger.warning("%s not found. Using existing environment variables.", env_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_PATH)
    parser.add_argument(
        "--model-name", default=os.getenv("MLFLOW_MODEL_NAME", "m5_forecasting_model")
    )
    parser.add_argument("--alias", default=os.getenv("MLFLOW_MODEL_ALIAS", "champion"))
    return parser.parse_args()


class PredictionRequest(BaseModel):
    """The three values needed to look up an already-engineered feature row."""

    item_id: str = Field(..., examples=["HOBBIES_1_001"])
    store_id: str = Field(..., examples=["CA_1"])
    day: str = Field(..., examples=["d_1886"])


class PredictionResponse(BaseModel):
    item_id: str
    store_id: str
    day: str
    prediction: float
    model_name: str
    model_version: str
    alias: str


model: Any | None = None
model_name = "m5_forecasting_model"
model_alias = "champion"
model_version = "unknown"
model_uri = ""
model_input_names: list[str] = []
model_store = "CA_1"
feature_rows: pd.DataFrame | None = None


def normalize_day(value: str | int) -> str:
    """Normalize day inputs such as 1886 or d_1886 to d_1886."""
    text = str(value).strip().lower().removeprefix("d_")
    if not text.isdigit():
        raise HTTPException(status_code=422, detail="day must look like d_1886 or 1886")
    return f"d_{int(text)}"


def load_feature_rows() -> None:
    """Load and validate the request-keyed feature table and model schema."""
    global feature_rows, model_input_names, model_store

    if not FEATURES_PATH.is_file():
        raise RuntimeError(f"Feature CSV not found: {FEATURES_PATH}")
    if not DEFAULT_META_PATH.is_file():
        raise RuntimeError(f"Model metadata not found: {DEFAULT_META_PATH}")

    metadata = json.loads(DEFAULT_META_PATH.read_text(encoding="utf-8"))
    model_input_names = metadata.get("features", [])
    model_store = metadata.get("store", "CA_1")
    if not model_input_names:
        raise RuntimeError(f"No feature schema found in {DEFAULT_META_PATH}")

    rows = pd.read_csv(FEATURES_PATH)
    missing = [name for name in [*LOOKUP_COLUMNS, *model_input_names] if name not in rows]
    if missing:
        raise RuntimeError(
            "Feature CSV is missing lookup/model columns: " + ", ".join(missing)
        )
    feature_rows = rows
    logger.info("Loaded %d feature rows with %d model features", len(rows), len(model_input_names))


def load_registered_model(name: str, alias: str) -> None:
    """Resolve the configured MLflow alias and load its LightGBM model."""
    global model, model_name, model_alias, model_version, model_uri

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise RuntimeError("MLFLOW_TRACKING_URI is missing from .env or environment.")

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    registered_version = client.get_model_version_by_alias(name=name, alias=alias)
    model_version = str(registered_version.version)
    model_uri = f"models:/{name}/{model_version}"
    logger.info("Loading registered model %s version %s (alias %s)", name, model_version, alias)

    loaded_model = mlflow.lightgbm.load_model(model_uri)
    if not isinstance(loaded_model, lgb.Booster):
        raise TypeError(f"Expected a LightGBM Booster, got {type(loaded_model).__name__}.")
    registered_features = loaded_model.feature_name()
    if model_input_names and registered_features != model_input_names:
        raise RuntimeError(
            "Registered model feature names/order do not match the local metadata. "
            "Check that the model alias points to the CA_1 cv4 model."
        )

    model = loaded_model
    model_name = name
    model_alias = alias


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_env_file(Path(os.getenv("MLFLOW_ENV_FILE", str(DEFAULT_ENV_PATH))))
    load_feature_rows()
    load_registered_model(
        os.getenv("MLFLOW_MODEL_NAME", "m5_forecasting_model"),
        os.getenv("MLFLOW_MODEL_ALIAS", "champion"),
    )
    yield


app = FastAPI(
    title="M5 Forecasting API",
    description=(
        "Send item_id, store_id, and day. The API looks up the matching processed "
        "feature row and predicts with the registered LightGBM model."
    ),
    version="1.1.0",
    lifespan=lifespan,
)


def _find_feature_row(request: PredictionRequest) -> pd.DataFrame:
    if feature_rows is None:
        raise HTTPException(status_code=503, detail="Feature data is not loaded")

    item_id = request.item_id.strip()
    store_id = request.store_id.strip()
    day = normalize_day(request.day)
    if store_id != model_store:
        raise HTTPException(
            status_code=422,
            detail=f"This registered model is for store {model_store}; got {store_id}.",
        )

    matches = feature_rows[
        (feature_rows["request_item_id"].astype(str) == item_id)
        & (feature_rows["request_store_id"].astype(str) == store_id)
        & (feature_rows["request_day"].astype(str).str.lower() == day)
    ]
    if matches.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No processed feature row for {item_id}, {store_id}, {day}. "
                "Add that request's engineered row to prediction_features.csv."
            ),
        )
    if len(matches) > 1:
        raise HTTPException(status_code=409, detail="Feature CSV has duplicate rows for this request.")

    frame = matches[model_input_names].copy()
    for column in CATEGORICAL_FEATURES:
        if column in frame:
            frame[column] = frame[column].astype("category")
    return frame


def _predict(request: PredictionRequest) -> PredictionResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    frame = _find_feature_row(request)
    try:
        result = model.predict(frame)
        prediction = float(result[0])
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=422,
            detail="Prediction failed. Check that feature values and categorical inputs match training.",
        ) from exc

    return PredictionResponse(
        item_id=request.item_id.strip(),
        store_id=request.store_id.strip(),
        day=normalize_day(request.day),
        prediction=prediction,
        model_name=model_name,
        model_version=model_version,
        alias=model_alias,
    )


@app.get("/")
def home() -> dict[str, str]:
    return {
        "message": "M5 Forecasting API",
        "model_name": model_name,
        "model_alias": model_alias,
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict[str, str]:
    if model is None or feature_rows is None:
        raise HTTPException(status_code=503, detail="Model or feature data is not loaded")
    return {"status": "healthy", "model_name": model_name, "model_version": model_version}


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return {
        "model_name": model_name,
        "model_version": model_version,
        "alias": model_alias,
        "model_uri": model_uri,
        "store": model_store,
        "expected_features": model_input_names,
        "feature_rows": 0 if feature_rows is None else len(feature_rows),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Predict from the three request fields using the matching CSV row."""
    return _predict(request)


@app.post("/predict-from-csv", response_model=PredictionResponse)
def predict_from_csv() -> PredictionResponse:
    """Predict from the first request row in prediction_requests.csv."""
    if not REQUESTS_PATH.is_file():
        raise HTTPException(status_code=404, detail=f"Request CSV not found: {REQUESTS_PATH}")
    try:
        requests = pd.read_csv(REQUESTS_PATH, dtype=str).fillna("")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Could not read prediction_requests.csv") from exc
    needed = ["item_id", "store_id", "day"]
    missing = [name for name in needed if name not in requests]
    if missing:
        raise HTTPException(status_code=400, detail=f"Request CSV missing columns: {missing}")
    if requests.empty or any(not str(requests.iloc[0][name]).strip() for name in needed):
        raise HTTPException(
            status_code=400,
            detail="Fill item_id, store_id, and day in the first row of prediction_requests.csv.",
        )
    row = requests.iloc[0]
    return _predict(
        PredictionRequest(
            item_id=str(row["item_id"]),
            store_id=str(row["store_id"]),
            day=str(row["day"]),
        )
    )


if __name__ == "__main__":
    import uvicorn

    args = parse_args()
    load_env_file(args.env_file)
    os.environ["MLFLOW_MODEL_NAME"] = args.model_name
    os.environ["MLFLOW_MODEL_ALIAS"] = args.alias
    os.environ["MLFLOW_ENV_FILE"] = str(args.env_file.resolve())
    uvicorn.run("m5_app:app", host="127.0.0.1", port=8000, reload=False)
