import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# For SHAP values
import shap

# ------------------------------------------------------
# Logger
# ------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


# ------------------------------------------------------
# Training and Evaluation Functions (temporal split)
# ------------------------------------------------------
def train_test_split(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform a temporal split: the last % of rows are the test set.
    This preserves chronological order (critical for time-series-like data).
    """
    n = len(df)
    split_index = int(n * (1 - test_size))

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    logger.info(f"Data split into {len(train_df)} train rows and {len(test_df)} test rows.")
    return train_df, test_df


# ------------------------------------------------------
# Train XGBoost Model
# ------------------------------------------------------
def train_xgb(X_train: pd.DataFrame, y_train: pd.Series) -> XGBRegressor:
    """Train an XGBoost regressor using predefined hyperparameters."""

    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="reg:squarederror"
    )

    logger.info(" Training XGBoost model...")
    model.fit(X_train, y_train)
    logger.info(" XGBoost training completed.")
    return model


# ------------------------------------------------------
# PREDICT ON TEST SET
# ------------------------------------------------------
def predict(model: XGBRegressor, X_test: pd.DataFrame) -> np.ndarray:
    """Generate predictions aligned with X_test index."""

    preds = model.predict(X_test)
    return preds


# ------------------------------------------------------
# Main function: Train and Evaluate
# ------------------------------------------------------
def train_and_evaluate(df_features: pd.DataFrame):
    """
    Full ML pipeline:
    - Split data into train/test
    - Train XGBoost
    - Predict on test set
    - Compute metrics
    - Compute SHAP values
    - Return all relevant outputs
    """
    logger.info(" Starting model training and evaluation...")

    # Identify target and features
    target = "mood_score"
    feature_cols = [col for col in df_features.columns if col not in ["date", target]]

    # Split
    train_df, test_df = train_test_split(df_features)

    X_train = train_df[feature_cols]
    y_train = train_df[target]

    X_test = test_df[feature_cols]
    y_test = test_df[target]

    # Train model
    model = train_xgb(X_train, y_train)

    # Predictions
    preds = predict(model, X_test)

    # Metrics
    mae = np.mean(np.abs(y_test - preds))
    rmse = np.sqrt(np.mean((y_test - preds) ** 2))


    logger.info(f" MAE: {mae:.4f}")
    logger.info(f" RMSE: {rmse:.4f}")

    # Build prediction DataFrame
    df_predictions = pd.DataFrame({
        "date": test_df["date"].values,
        "true_mood": y_test.values,
        "predicted_mood": preds
    })

    # SHAP values
    logger.info("Computing SHAP values for model interpretability...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    shap_df = pd.DataFrame(shap_values, columns=feature_cols)
    shap_df.insert(0, "date", test_df["date"].values)

    logger.info("Model evaluation + SHAP computation complete.")

    # Return all
    metrics = {"mae": mae, "rmse": rmse} # mean absolute error, root mean squared error

    return model, metrics, df_predictions, shap_df
