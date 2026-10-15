"""Train a non-recursive LightGBM model for one M5 store.

Expected input files under data/processed/:
    grid_part_1.pkl
    grid_part_2.pkl
    grid_part_3.pkl
    lags_df_28.pkl
    mean_encoding_df.pkl

Example:
    python src/model/model_building.py
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import pickle
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

TARGET = "sales"
PROJECT_DIR = Path.cwd()
DATA_DIR = PROJECT_DIR / "data" / "processed"
MODEL_DIR = PROJECT_DIR / "models"

# (train_end_day, valid_end_day): validate on days in (train_end, valid_end].
# Private days d_1942..d_1969 do not have labels in the public training file.
FOLDS: dict[str, tuple[int, int]] = {
    "cv1": (1551, 1610),
    "cv2": (1829, 1857),
    "cv3": (1857, 1885),
    "cv4": (1885, 1913),
    "public": (1913, 1941),
}

REMOVE_FEATURES = ["id", "state_id", "store_id", "date", "wm_yr_wk", "d", TARGET]

GRID2_COLS = [
    "sell_price",
    "price_max",
    "price_min",
    "price_std",
    "price_mean",
    "price_norm",
    "price_nunique",
    "item_nunique",
    "price_momentum",
    "price_momentum_m",
    "price_momentum_y",
]
GRID3_COLS = [
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
    "snap_CA",
    "snap_TX",
    "snap_WI",
    "tm_d",
    "tm_w",
    "tm_m",
    "tm_y",
    "tm_wm",
    "tm_dw",
    "tm_w_end",
    "day_of_month",
    "week_of_year",
    "month",
    "year",
    "day_of_week",
    "is_weekend",
]
LAG_COLS = (
    [f"sales_lag_{lag}" for lag in range(28, 43)]
    + [f"rolling_{stat}_{window}" for window in (7, 14, 30, 60, 180) for stat in ("mean", "std")]
)
MEAN_ENC_COLS = [
    "enc_store_id_dept_id_mean",
    "enc_store_id_dept_id_std",
    "enc_item_id_state_id_mean",
    "enc_item_id_state_id_std",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("model_building")


def load_params(params_path: Path) -> dict[str, Any]:
    """Load model-building parameters from params.yaml."""
    with params_path.open("r", encoding="utf-8") as file:
        params = yaml.safe_load(file) or {}

    model_params = params.get("model_building")
    if not isinstance(model_params, dict):
        raise TypeError(f"{params_path} must contain a 'model_building' section.")
    return model_params


def get_config(args: argparse.Namespace) -> dict[str, Any]:
    """Merge params.yaml values with explicit command-line overrides."""
    config = load_params(Path(args.params))

    overrides = {
        "data_dir": args.data_dir,
        "model_dir": args.model_dir,
        "store": args.store,
        "cv": args.cv,
        "first_day": args.first_day,
        "early_stopping": args.early_stopping,
        "max_items": args.max_items,
        "num_threads": args.num_threads,
        "num_boost_round": args.num_boost_round,
        "force": args.force,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value

    config.setdefault("data_dir", DATA_DIR)
    config.setdefault("model_dir", MODEL_DIR)
    config.setdefault("store", "CA_1")
    config.setdefault("cv", "cv4")
    config.setdefault("first_day", 710)
    config.setdefault("early_stopping", 50)
    config.setdefault("max_items", None)
    config.setdefault("num_threads", 0)
    config.setdefault("num_boost_round", 1500)
    config.setdefault("force", False)
    config.setdefault("lgb_params", {})

    if config["cv"] not in FOLDS:
        raise ValueError(f"cv must be one of {sorted(FOLDS)}.")
    if not isinstance(config["lgb_params"], dict):
        raise TypeError("model_building.lgb_params must be a dictionary.")

    return config


def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns and convert object columns to categories."""
    for column in df.columns:
        dtype = df[column].dtype
        if pd.api.types.is_integer_dtype(dtype):
            df[column] = pd.to_numeric(df[column], downcast="integer")
        elif dtype == np.float64:
            df[column] = df[column].astype(np.float32)
        elif dtype == object or pd.api.types.is_string_dtype(dtype):
            df[column] = df[column].astype("category")
    return df


def pick_existing_columns(wanted: list[str], available: pd.Index, feature_group: str) -> list[str]:
    """Return columns that exist and log any optional columns that are missing."""
    present = [column for column in wanted if column in available]
    missing = [column for column in wanted if column not in available]
    if missing:
        logger.warning(
            "%s: %d expected columns not found, skipping: %s",
            feature_group,
            len(missing),
            missing,
        )
    return present


def load_subset(
    path: Path,
    wanted_columns: list[str],
    rows: pd.Index,
    full_index: pd.Index,
    feature_group: str,
) -> pd.DataFrame:
    """Load a processed pickle and keep only aligned rows and requested columns."""
    df = pd.read_pickle(path)
    if not df.index.equals(full_index):
        raise ValueError(
            f"{path.name} has a different row index from grid_part_1.pkl. "
            "Rows cannot be aligned safely."
        )

    subset = df.loc[rows, pick_existing_columns(wanted_columns, df.columns, feature_group)]
    del df
    gc.collect()
    return subset


def load_store_frame(
    data_dir: Path,
    store: str,
    first_day: int,
    max_items: int | None,
) -> pd.DataFrame:
    """Build the training table for one store from the preprocessed feature parts."""
    grid1 = pd.read_pickle(data_dir / "grid_part_1.pkl")
    full_index = grid1.index

    keep = (grid1["store_id"] == store) & (grid1["d"] >= first_day)
    if not keep.any():
        raise ValueError(f"No rows found for store {store!r} with d >= {first_day}.")

    base = grid1.loc[keep]
    del grid1
    gc.collect()

    if max_items:
        ids = base["id"].drop_duplicates().iloc[:max_items]
        base = base[base["id"].isin(ids)]
        logger.info("Using the first %d series only.", len(ids))

    rows = base.index
    parts = [
        base,
        load_subset(data_dir / "grid_part_2.pkl", GRID2_COLS, rows, full_index, "price features"),
        load_subset(data_dir / "grid_part_3.pkl", GRID3_COLS, rows, full_index, "calendar features"),
        load_subset(data_dir / "lags_df_28.pkl", LAG_COLS, rows, full_index, "lag features"),
        load_subset(
            data_dir / "mean_encoding_df.pkl",
            MEAN_ENC_COLS,
            rows,
            full_index,
            "mean encodings",
        ),
    ]
    df = pd.concat(parts, axis=1)
    del parts, base
    gc.collect()

    df = reduce_mem_usage(df)
    memory_mb = df.memory_usage(deep=True).sum() / 1e6
    logger.info("Loaded %s: %d rows x %d columns, %.0f MB", store, len(df), df.shape[1], memory_mb)
    return df


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate simple regression metrics."""
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    error = y_pred - y_true
    total = float(np.abs(y_true).sum())
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "wape": float(np.abs(error).sum() / total) if total > 0 else float("nan"),
    }


def train_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    params: dict[str, Any],
    num_boost_round: int,
    early_stopping: int,
) -> tuple[lgb.Booster, dict[str, Any], float, int]:
    """Train the LightGBM model and return training details."""
    dtrain = lgb.Dataset(x_train, label=y_train)
    dvalid = lgb.Dataset(x_valid, label=y_valid, reference=dtrain)
    callbacks: list[Callable[..., Any]] = [lgb.log_evaluation(50)]
    if early_stopping > 0:
        callbacks.append(lgb.early_stopping(early_stopping, verbose=True))

    start = time.perf_counter()
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dvalid],
        valid_names=["valid"],
        callbacks=callbacks,
    )
    train_seconds = time.perf_counter() - start
    best_iteration = model.best_iteration or model.current_iteration()
    logger.info("Training took %.1f seconds, best iteration %d", train_seconds, best_iteration)
    return model, params, train_seconds, best_iteration


def save_outputs(
    model: lgb.Booster,
    model_dir: Path,
    stem: str,
    best_iteration: int,
    valid_ids: pd.DataFrame,
    y_valid: np.ndarray,
    prediction: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    """Save model, validation predictions, and run metadata."""
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_dir / f"{stem}.txt"), num_iteration=best_iteration)
    with (model_dir / "model.pkl").open("wb") as file:
        pickle.dump(model, file)

    predictions = valid_ids.assign(
        y_true=y_valid,
        y_pred=prediction,
    )
    predictions.to_csv(model_dir / f"{stem}_predictions.csv", index=False)
    (model_dir / f"{stem}_meta.json").write_text(json.dumps(metadata, indent=2))
    logger.info("Saved model, predictions, and metadata to %s", model_dir)


def run_training(config: dict[str, Any]) -> None:
    """Run the full model-building flow."""
    cutoff, end = FOLDS[config["cv"]]
    model_dir = Path(config["model_dir"])
    data_dir = Path(config["data_dir"])
    stem = f"non_recur_{config['store']}_{config['cv']}"
    model_path = model_dir / f"{stem}.txt"

    if model_path.exists() and not config["force"]:
        logger.info("%s already exists. Use --force to retrain.", model_path)
        return

    df = load_store_frame(data_dir, config["store"], config["first_day"], config["max_items"])
    day = df["d"]
    train_mask = (day >= config["first_day"]) & (day <= cutoff) & df[TARGET].notna()
    valid_mask = (day > cutoff) & (day <= end)

    if df.loc[valid_mask, TARGET].isna().any():
        raise ValueError(f"Fold {config['cv']} has missing sales in its validation window.")
    if not train_mask.any() or not valid_mask.any():
        raise ValueError("Empty train or validation split. Check --first-day and --cv.")

    features = [column for column in df.columns if column not in REMOVE_FEATURES]
    logger.info(
        "Train days %d..%d (%d rows) | validate days %d..%d (%d rows) | %d features",
        config["first_day"],
        cutoff,
        int(train_mask.sum()),
        cutoff + 1,
        end,
        int(valid_mask.sum()),
        len(features),
    )

    y_valid_np = df.loc[valid_mask, TARGET].to_numpy()
    x_train = df.loc[train_mask, features]
    y_train = df.loc[train_mask, TARGET]
    x_valid = df.loc[valid_mask, features]
    y_valid = df.loc[valid_mask, TARGET]
    valid_ids = df.loc[valid_mask, ["id", "d"]].copy()

    del df
    gc.collect()

    lgb_params = dict(config["lgb_params"])
    if config["num_threads"]:
        lgb_params["num_threads"] = config["num_threads"]

    model, params, train_seconds, best_iteration = train_model(
        x_train,
        y_train,
        x_valid,
        y_valid,
        lgb_params,
        config["num_boost_round"],
        config["early_stopping"],
    )

    prediction = np.asarray(model.predict(x_valid), dtype=np.float64)
    metrics = regression_metrics(y_valid_np, prediction)
    logger.info("Model metrics: %s", {key: round(value, 4) for key, value in metrics.items()})

    metadata = {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "store": config["store"],
        "fold": config["cv"],
        "train_days": [config["first_day"], cutoff],
        "valid_days": [cutoff + 1, end],
        "train_rows": len(x_train),
        "valid_rows": len(x_valid),
        "max_items": config["max_items"],
        "params": params,
        "best_iteration": int(best_iteration),
        "train_seconds": round(train_seconds, 1),
        "features": features,
        "metrics": metrics,
        "lightgbm_version": lgb.__version__,
    }
    save_outputs(
        model,
        model_dir,
        stem,
        best_iteration,
        valid_ids,
        y_valid_np,
        prediction,
        metadata,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, default=PROJECT_DIR / "params.yaml")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--store", default=None)
    parser.add_argument("--cv", choices=sorted(FOLDS), default=None)
    parser.add_argument("--first-day", type=int, default=None)
    parser.add_argument("--early-stopping", type=int, default=None)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument("--num-boost-round", type=int, default=None)
    parser.add_argument("--force", action="store_true", default=None)
    return parser.parse_args()


def main() -> None:
    """Script entry point."""
    run_training(get_config(parse_args()))


if __name__ == "__main__":
    main()
