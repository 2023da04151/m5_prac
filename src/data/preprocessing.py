"""Simple, beginner-friendly preprocessing for the M5 forecasting project.

Expected input files:
    data/raw/sales_train_evaluation.csv
    data/raw/sell_prices.csv
    data/raw/calendar.csv

Run from the project root:
    python data_preprocessing.py

This creates intermediate pickle files under data/processed/.
The full M5 dataset is large, so run this with enough RAM.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# 1. Project settings
# ---------------------------------------------------------------------
PROJECT_DIR = Path.cwd()
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

TARGET = "sales"
LAST_TRAIN_DAY = 1941
FORECAST_HORIZON = 28
LAG_SHIFT = 28

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# 2. Small helper functions
# ---------------------------------------------------------------------
def reduce_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Use smaller numeric types where possible to reduce RAM usage."""
    for column in df.columns:
        if pd.api.types.is_integer_dtype(df[column]):
            df[column] = pd.to_numeric(df[column], downcast="integer")
        elif df[column].dtype == "float64":
            df[column] = df[column].astype("float32")
    return df


def save_pickle(df: pd.DataFrame, filename: str) -> None:
    """Save a dataframe so later steps can load it without rebuilding it."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / filename
    df.to_pickle(path)
    logger.info("Saved %s | shape=%s", path, df.shape)


def load_source_data():
    """Read the three original M5 CSV files."""
    logger.info("Loading M5 CSV files")
    sales = pd.read_csv(RAW_DIR / "sales_train_evaluation.csv")
    prices = pd.read_csv(RAW_DIR / "sell_prices.csv")
    calendar = pd.read_csv(RAW_DIR / "calendar.csv")
    return sales, prices, calendar


# ---------------------------------------------------------------------
# 3. Create the main daily sales table (grid_part_1)
# ---------------------------------------------------------------------
def create_sales_grid(sales: pd.DataFrame, prices: pd.DataFrame,
                      calendar: pd.DataFrame) -> pd.DataFrame:
    """Convert wide sales data into one row per item-store-day."""
    id_columns = [
        "id", "item_id", "dept_id", "cat_id", "store_id", "state_id"
    ]
    logger.info("Converting daily sales from wide format to long format")

    # d_1, d_2, ... become rows in the new 'd' column.
    grid = sales.melt(
        id_vars=id_columns,
        var_name="d",
        value_name=TARGET,
    )
    grid["d"] = grid["d"].str.replace("d_", "", regex=False).astype("int16")

    # Add 28 future rows per item-store series for forecasting.
    series = sales[id_columns].drop_duplicates()
    future_rows = []
    for day in range(LAST_TRAIN_DAY + 1, LAST_TRAIN_DAY + FORECAST_HORIZON + 1):
        day_frame = series.copy()
        day_frame["d"] = day
        day_frame[TARGET] = np.nan
        future_rows.append(day_frame)
    grid = pd.concat([grid, *future_rows], ignore_index=True)

    # Find the first week in which each item was listed in a store.
    release = prices.groupby(["store_id", "item_id"], as_index=False).agg(
        release_week=("wm_yr_wk", "min")
    )
    grid = grid.merge(release, on=["store_id", "item_id"], how="left")

    # Map each day number to its calendar week.
    day_to_week = calendar[["d", "wm_yr_wk"]].copy()
    day_to_week["d"] = day_to_week["d"].str.replace("d_", "", regex=False).astype("int16")
    grid = grid.merge(day_to_week, on="d", how="left")

    # Drop rows before an item was listed in that store.
    grid = grid[grid["wm_yr_wk"] >= grid["release_week"]].copy()
    grid = grid.drop(columns=["release_week", "wm_yr_wk"])
    grid = grid.reset_index(drop=True)

    # Repeated IDs take less memory as categorical columns.
    for column in ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]:
        grid[column] = grid[column].astype("category")

    return reduce_memory(grid)


# ---------------------------------------------------------------------
# 4. Create price features (grid_part_2)
# ---------------------------------------------------------------------
def create_price_features(prices: pd.DataFrame,
                          calendar: pd.DataFrame) -> pd.DataFrame:
    """Create summary and relative-price features for each item and store."""
    logger.info("Creating price features")
    prices = prices.copy()

    group = prices.groupby(["store_id", "item_id"])["sell_price"]
    prices["price_max"] = group.transform("max")
    prices["price_min"] = group.transform("min")
    prices["price_mean"] = group.transform("mean")
    prices["price_std"] = group.transform("std")
    prices["price_norm"] = prices["sell_price"] / prices["price_max"]
    prices["price_nunique"] = group.transform("nunique")

    # Sort first so shift(1) means the previous week for that item-store.
    prices = prices.sort_values(["store_id", "item_id", "wm_yr_wk"])
    prices["price_momentum"] = (
        prices["sell_price"]
        / prices.groupby(["store_id", "item_id"])["sell_price"].shift(1)
    )

    # Keep only the columns needed to join price features to daily rows.
    feature_columns = [
        "store_id", "item_id", "wm_yr_wk", "sell_price",
        "price_max", "price_min", "price_mean", "price_std",
        "price_norm", "price_nunique", "price_momentum",
    ]
    return reduce_memory(prices[feature_columns])


def create_calendar_features(calendar: pd.DataFrame) -> pd.DataFrame:
    """Create day, week, month, year, event and SNAP features."""
    logger.info("Creating calendar features")
    calendar = calendar.copy()
    calendar["d"] = calendar["d"].str.replace("d_", "", regex=False).astype("int16")
    calendar["date"] = pd.to_datetime(calendar["date"])

    calendar["day_of_month"] = calendar["date"].dt.day.astype("int8")
    calendar["week_of_year"] = calendar["date"].dt.isocalendar().week.astype("int16")
    calendar["month"] = calendar["date"].dt.month.astype("int8")
    calendar["year"] = (calendar["date"].dt.year - calendar["date"].dt.year.min()).astype("int8")
    calendar["day_of_week"] = calendar["date"].dt.dayofweek.astype("int8")
    calendar["is_weekend"] = (calendar["day_of_week"] >= 5).astype("int8")

    columns = [
        "d", "event_name_1", "event_type_1", "event_name_2", "event_type_2",
        "snap_CA", "snap_TX", "snap_WI", "day_of_month", "week_of_year",
        "month", "year", "day_of_week", "is_weekend",
    ]
    return calendar[columns]


# ---------------------------------------------------------------------
# 5. Create lag and rolling features
# ---------------------------------------------------------------------
def create_lag_features(grid: pd.DataFrame) -> pd.DataFrame:
    """Create historical sales features for each item-store series."""
    logger.info("Creating lag and rolling features")
    result = grid[["id", "d", TARGET]].copy()
    grouped_sales = result.groupby("id", observed=True)[TARGET]

    # A 28-day lag keeps the target horizon out of the feature.
    for lag in range(LAG_SHIFT, LAG_SHIFT + 15):
        result[f"sales_lag_{lag}"] = grouped_sales.shift(lag).astype("float32")

    # Rolling summaries use only sales at least 28 days in the past.
    for window in [7, 14, 30, 60, 180]:
        shifted_sales = result.groupby("id", observed=True)[TARGET].shift(LAG_SHIFT)
        rolling = shifted_sales.groupby(result["id"], observed=True).rolling(window)
        result[f"rolling_mean_{window}"] = (
            rolling.mean().reset_index(level=0, drop=True).sort_index().astype("float32")
        )
        result[f"rolling_std_{window}"] = (
            rolling.std().reset_index(level=0, drop=True).sort_index().astype("float32")
        )

    return result.drop(columns=[TARGET])


# ---------------------------------------------------------------------
# 6. Create group-level mean and standard-deviation features
# ---------------------------------------------------------------------
def create_mean_encodings(grid: pd.DataFrame) -> pd.DataFrame:
    """Attach historical group sales averages and standard deviations."""
    logger.info("Creating group mean/std features")
    data = grid.copy()

    # Do not use sales from the last 28 training days or future days
    # when calculating these encodings for this validation setup.
    data.loc[data["d"] > LAST_TRAIN_DAY - FORECAST_HORIZON, TARGET] = np.nan

    groupings = [
        ["state_id"], ["store_id"], ["cat_id"], ["dept_id"],
        ["state_id", "cat_id"], ["state_id", "dept_id"],
        ["store_id", "cat_id"], ["store_id", "dept_id"],
        ["item_id"], ["item_id", "state_id"], ["item_id", "store_id"],
    ]

    encoded = data[["id", "d"]].copy()
    for columns in groupings:
        name = "_".join(columns)
        grouped = data.groupby(columns, observed=True)[TARGET]
        encoded[f"enc_{name}_mean"] = grouped.transform("mean").astype("float32")
        encoded[f"enc_{name}_std"] = grouped.transform("std").astype("float32")

    return encoded


# ---------------------------------------------------------------------
# 7. Run the preprocessing pipeline
# ---------------------------------------------------------------------
def main() -> None:
    sales, prices, calendar = load_source_data()

    grid = create_sales_grid(sales, prices, calendar)
    save_pickle(grid, "grid_part_1.pkl")

    price_features = create_price_features(prices, calendar)
    day_to_week = calendar[["d", "wm_yr_wk"]].copy()
    day_to_week["d"] = day_to_week["d"].str.replace("d_", "", regex=False).astype("int16")
    grid_with_week = grid.merge(day_to_week, on="d", how="left")
    price_features = grid_with_week[["id", "d", "store_id", "item_id", "wm_yr_wk"]].merge(
        price_features, on=["store_id", "item_id", "wm_yr_wk"], how="left"
    )
    price_features = price_features.drop(
        columns=["store_id", "item_id", "wm_yr_wk", "sell_price"], errors="ignore"
    )
    save_pickle(price_features, "grid_part_2.pkl")
    del grid_with_week, price_features

    calendar_features = create_calendar_features(calendar)
    calendar_features = grid[["id", "d"]].merge(calendar_features, on="d", how="left")
    save_pickle(calendar_features, "grid_part_3.pkl")
    del calendar_features

    lag_features = create_lag_features(grid)
    save_pickle(lag_features, "lags_df_28.pkl")
    del lag_features

    mean_encodings = create_mean_encodings(grid)
    save_pickle(mean_encodings, "mean_encoding_df.pkl")

    logger.info("Preprocessing complete. Files saved in %s", PROCESSED_DIR)


if __name__ == "__main__":
    main()
