import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import random

# ------------------------------------------------------
# global logger configuration
# ------------------------------------------------------
def get_logger(name: str) -> logging.Logger:
    """
    Create and configure a logger with unified formatting.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


# ------------------------------------------------------
# Seed Setting Function 
# ------------------------------------------------------
def set_seed(seed: int = 42):
    """
    Ensure full reproducibility across:
    - numpy
    - random
    - (optional) XGBoost
    """
    random.seed(seed)
    np.random.seed(seed)


# ------------------------------------------------------
# Creation of directories
# ------------------------------------------------------
def ensure_directory(path: str):
    """
    Create a directory if it does not exist.
    """
    Path(path).mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------
# Save DataFrame to CSV
# ------------------------------------------------------
def save_csv(df: pd.DataFrame, path: str):
    """
    Save a DataFrame to CSV safely and ensure directory exists.
    """
    path_obj = Path(path)
    ensure_directory(path_obj.parent)
    df.to_csv(path_obj, index=False)
