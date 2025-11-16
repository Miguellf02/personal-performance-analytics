import pandas as pd
import numpy as np
import shap
import logging
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------------------------------------------
# Loggers 
# ------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


# ------------------------------------------------------
# Feature Importance Plot
# ------------------------------------------------------
def plot_feature_importance(model, feature_names, save_path="graphics/feature_importance.png"):
    """Generate and save a feature importance bar plot based on the trained XGBoost model"""

    logger.info(" Generating feature importance plot...")

    importances = model.feature_importances_

    # Sort features by importance
    sorted_idx = np.argsort(importances)
    sorted_names = np.array(feature_names)[sorted_idx]
    sorted_values = importances[sorted_idx]

    plt.figure(figsize=(8, 10))
    plt.barh(sorted_names, sorted_values)
    plt.title("XGBoost Feature Importance")
    plt.xlabel("Importance Score")
    plt.tight_layout()

    # Ensure directory exists
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

    logger.info(f" Feature importance saved to: {save_path}")


# ------------------------------------------------------
# Shap Summary Plot
# ------------------------------------------------------
def plot_shap_summary(shap_values, X_test, save_path="graphics/shap_summary.png"):
    """ Generate a SHAP summary plot (global interpretability). """

    logger.info(" Generating SHAP summary plot...")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    logger.info(f" SHAP summary plot saved to: {save_path}")


# ------------------------------------------------------
# Shap Dependence Plot
# ------------------------------------------------------
def plot_shap_dependence(shap_values, X_test, feature, save_path=None):
    """
    Generate a SHAP dependence plot for a single feature.
    This function is optional and not required for the main pipeline.
    """
    logger.info(f" Generating SHAP dependence plot for {feature}...")

    if save_path is None:
        save_path = f"graphics/shap_dependence_{feature}.png"

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    shap.dependence_plot(feature, shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    logger.info(f"SHAP dependence plot saved to: {save_path}")
