import pandas as pd
import numpy as np
import logging
from pathlib import Path

# ------------------------------------------------------
# LOGGER CONFIGURATION
# ------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


# ------------------------------------------------------
# Internal preprocessing functions
# ------------------------------------------------------

def _convert_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """ Convert "-" symbols to proper NaN values. """
    df = df.replace("-", np.nan)
    return df


def _fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """ Convert columns into correct data types. """

    numeric_cols = [
        "horas_sueno", "calorias", "fuerza", "cardio_min",
        "pasos", "peso", "estres", "alcohol", "mood_score"
    ]
    
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    return df


def _clip_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Clip extreme outliers to reasonable physiological ranges."""

    df["horas_sueno"] = df["horas_sueno"].clip(4, 10) 
    df["calorias"] = df["calorias"].clip(1300, 4000)
    df["cardio_min"] = df["cardio_min"].clip(0, 120)
    df["peso"] = df["peso"].clip(60, 100)
    df["estres"] = df["estres"].clip(1, 5)
    df["alcohol"] = df["alcohol"].clip(0, 5)
    
    return df


def _impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """  Impute missing values using forward-fill and backward-fill.
    This guarantees a complete dataset while keeping realistic transitions."""

    df = df.ffill().bfill() 
    return df


# ------------------------------------------------------
# Main function: Full preprocessing pipeline
# ------------------------------------------------------

def preprocess(df_raw: pd.DataFrame, save_path: str = "data/processed/metrics_clean.csv") -> pd.DataFrame:
    """
    Full preprocessing pipeline:
    - Convert missing symbols
    - Fix data types
    - Impute missing values
    - Clip outliers
    - Save cleaned dataset
    """
    logger.info(" Starting preprocessing pipeline...")

    df = df_raw.copy()

    # Step 1 — Replace '-' with NaN
    df = _convert_missing_values(df)
    logger.info("Step 1/4: Converted '-' to NaN.")

    # Step 2 — Convert column types
    df = _fix_dtypes(df)
    logger.info("Step 2/4: Fixed numeric data types.")

    # Step 3 — Impute missing values
    df = _impute_missing(df)
    logger.info("Step 3/4: Missing values imputed with ffill/bfill.")

    # Step 4 — Clip unreasonable values
    df = _clip_outliers(df)
    logger.info("Step 4/4: Outliers clipped.")

    # Save cleaned file
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    logger.info(f" Clean dataset saved to: {save_path}")

    return df
