import logging
from pathlib import Path

from data_loader import load_raw_data
from preprocessing import preprocess
from feature_engineering import create_features
from model import train_and_evaluate
from evaluation import (
    plot_feature_importance,
    plot_shap_summary
)
from utils import get_logger, ensure_directory, save_csv, set_seed


# ------------------------------------------------------
# Main Pipeline Function
# ------------------------------------------------------
def main():
    logger = get_logger(__name__)
    set_seed(42)

    logger.info(" Starting full pipeline...")

    # -------------------------------
    # 1) Load raw dataset
    # -------------------------------
    raw_path = "data/raw/metrics_raw.csv"
    df_raw = load_raw_data(raw_path)

    # -------------------------------
    # 2) Preprocess dataset
    # -------------------------------
    df_clean = preprocess(df_raw)

    # -------------------------------
    # 3) Feature engineering
    # -------------------------------
    df_features = create_features(df_clean)

    # -------------------------------
    # 4) Train model + evaluate
    # -------------------------------
    model, metrics, df_predictions, shap_df = train_and_evaluate(df_features)

    logger.info(f" Final metrics: {metrics}")

    # -------------------------------
    # 5) EXPORT CSVs FOR POWER BI
    # -------------------------------
    ensure_directory("data/export")

    save_csv(df_features, "data/export/final_metrics.csv")
    save_csv(df_predictions, "data/export/predictions.csv")
    save_csv(shap_df, "data/export/shap_values.csv")

    logger.info(" Exported: final_metrics, predictions, shap_values CSVs")

    # -------------------------------
    # 6) GENERATE GRAPHICS
    # -------------------------------
    feature_names = [col for col in df_features.columns if col not in ["date", "mood_score"]]

    plot_feature_importance(model, feature_names, save_path="graphics/feature_importance.png")
    plot_shap_summary(shap_df.iloc[:, 1:].values, df_features[feature_names].iloc[-len(shap_df):], 
                      save_path="graphics/shap_summary.png")

    logger.info(" Pipeline completed successfully. Data ready for Power BI!")


# ------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------
if __name__ == "__main__":
    main()
