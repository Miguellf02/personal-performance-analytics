import pandas as pd
from typing import List
import logging

# ------------------------------------------------------
# We are going to set up logging for this module
# ------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


# ------------------------------------------------------
# Required columns in the raw CSV
# ------------------------------------------------------
REQUIRED_COLUMNS: List[str] = [
    "date",
    "horas_sueno",
    "calorias",
    "fuerza",
    "cardio_min",
    "pasos",
    "peso",
    "estres",
    "alcohol",
    "mood_score"
]


# ------------------------------------------------------
# Validation of raw CSV columns
# ------------------------------------------------------
def validate_raw_columns(df: pd.DataFrame):
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing_cols:
        logger.error(f"There are missing columns: {missing_cols}")
        raise ValueError(f"Missing columns: {missing_cols}")

    if list(df.columns) != REQUIRED_COLUMNS:
        logger.warning("⚠️ Las columnas existen pero NO están en el orden esperado. "
                       "Esto no afecta a pandas pero se recomienda corregirlo.")


# ------------------------------------------------------
# Load raw data from CSV
# ------------------------------------------------------
def load_raw_data(path: str) -> pd.DataFrame:
    """ Loads the raw dataset from a CSV file, validates columns, and parses dates."""
    try:
        logger.info(f"Loading raw dataset from: {path}")
        df = pd.read_csv(path)
    except FileNotFoundError:
        logger.error(f"Could not find file at {path}")
        raise

    
    validate_raw_columns(df)

    try:
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    except Exception as e:
        logger.error(f"An error occurred while converting 'date' column to datetime: {e}")
        raise

    # Log de valores faltantes
    missing_summary = df.isna().sum()
    if missing_summary.any():
        logger.warning("Missing values detected in the dataset:")
        logger.warning(f"\n{missing_summary}")

    logger.info("Successfully loaded and validated raw dataset.")
    logger.info(f"Shape: {df.shape}")

    return df
