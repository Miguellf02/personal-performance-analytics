import pandas as pd
import numpy as np
import logging
from pathlib import Path

# ------------------------------------------------------
# Logger configuration
# ------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


# ------------------------------------------------------
# Functions 
# ------------------------------------------------------

def _rolling_features(df: pd.DataFrame) -> pd.DataFrame: 
    """Create rolling window features (7-day moving averages).
    These capture temporal trends and smooth short-term noise."""

    df["sleep_7d"] = df["horas_sueno"].rolling(window=7).mean()
    df["calories_7d"] = df["calorias"].rolling(window=7).mean()
    df["stress_7d"] = df["estres"].rolling(window=7).mean()
    df["training_consistency"] = df["fuerza"].rolling(7).mean()

    return df


def _trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create trend-based features such as weight momentum and recovery index."""

    # Weight trend: rolling difference across 7 days
    df["weight_trend"] = df["peso"].diff(periods=7)

    # Cardio / calorie ratio (effort relative to intake)
    df["cardio_calorie_ratio"] = df["cardio_min"] / df["calorias"]

    # Recovery index (sleep vs stress balance)
    df["recovery_index"] = df["horas_sueno"] * (1 - df["estres"] / 5)

    return df


def _temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create temporal features such as day of week and weekend indicators."""

    df["day_of_week"] = df["date"].dt.weekday  # 0 = Monday, 6 = Sunday
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    return df


# ------------------------------------------------------
# Main function 
# ------------------------------------------------------

def create_features(df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pipeline:
    - Rolling averages
    - Trend indicators
    - Temporal categorical features
    - Final cleanup (drop initial NaNs from rolling windows)
    """
    logger.info("🔧 Starting feature engineering...")

    df = df_clean.copy()

    # Step 1 — Rolling windows
    df = _rolling_features(df)
    logger.info("Step 1/3: Rolling window features added.")

    # Step 2 — Trend-based features
    df = _trend_features(df)
    logger.info("Step 2/3: Trend and ratio features added.")

    # Step 3 — Temporal features
    df = _temporal_features(df)
    logger.info("Step 3/3: Temporal features added.")

    # Drop rows that contain NaN due to rolling-window initialization
    initial_rows = df.shape[0]
    df = df.dropna().reset_index(drop=True)
    removed = initial_rows - df.shape[0]

    logger.info(f" {removed} initial rows removed due to rolling window NaNs.")
    logger.info(" Feature engineering complete.")

    return df
