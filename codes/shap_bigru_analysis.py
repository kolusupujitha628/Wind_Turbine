import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import tensorflow as tf
import shap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "datasets"
RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_PATH = DATASET_DIR / "bo_bigru_anomaly_model.keras"
X_TEST_PATH = DATASET_DIR / "X_test_seq.npy"

CSV_PATH = RESULTS_DIR / "bo_bigru_shap_feature_importance.csv"
IMAGE_PATH = RESULTS_DIR / "bo_bigru_shap_feature_importance.png"

RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

SEED = 42
BACKGROUND_SAMPLES = 100
EXPLAIN_SAMPLES = 200


# ============================================================
# SCADA FEATURE NAMES
# ============================================================

FEATURE_NAMES = [
    "Gearbox Oil Temperature",
    "Gearbox Bearing Temperature",
    "Vibration X",
    "Vibration Y",
    "Vibration Z",
    "Oil Pressure",
    "Particle Count"
]


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print("\n========== BO-BiGRU SHAP EXPLAINABILITY ANALYSIS ==========")

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"BO-BiGRU model not found: {MODEL_PATH}"
        )

    X_test = np.load(
        X_TEST_PATH
    ).astype("float32")

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print("BO-BiGRU model loaded successfully.")
    print("Test data shape:", X_test.shape)

    # ========================================================
    # SELECT SAMPLES
    # ========================================================

    rng = np.random.default_rng(SEED)

    background_indices = rng.choice(
        len(X_test),
        size=min(BACKGROUND_SAMPLES, len(X_test)),
        replace=False
    )

    explain_indices = rng.choice(
        len(X_test),
        size=min(EXPLAIN_SAMPLES, len(X_test)),
        replace=False
    )

    X_background = X_test[background_indices]
    X_explain = X_test[explain_indices]

    print("Background samples:", X_background.shape)
    print("Explain samples   :", X_explain.shape)

    # ========================================================
    # PREDICTION
    # ========================================================

    predictions = model.predict(
        X_explain,
        verbose=0
    ).reshape(-1)

    predicted_anomalies = np.sum(
        predictions >= 0.5
    )

    print(
        "Predicted anomalies in SHAP sample:",
        predicted_anomalies
    )

    # ========================================================
    # SHAP ANALYSIS
    # ========================================================

    print("\nCalculating SHAP values... Please wait.")

    explainer = shap.GradientExplainer(
        model,
        X_background
    )

    shap_values = explainer.shap_values(
        X_explain
    )

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_values = np.array(shap_values)

    if shap_values.ndim == 4:
        shap_values = shap_values[..., 0]

    if shap_values.ndim != 3:
        raise ValueError(
            f"Unexpected SHAP output shape: {shap_values.shape}"
        )

    print("SHAP values shape:", shap_values.shape)

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    feature_importance = np.mean(
        np.abs(shap_values),
        axis=(0, 1)
    )

    importance_df = pd.DataFrame({
        "Feature": FEATURE_NAMES,
        "Mean_Absolute_SHAP_Value": feature_importance
    })

    importance_df = importance_df.sort_values(
        by="Mean_Absolute_SHAP_Value",
        ascending=False
    ).reset_index(drop=True)

    print("\n========== BO-BiGRU FEATURE IMPORTANCE ==========")

    print(
        importance_df.to_string(
            index=False
        )
    )

    # ========================================================
    # SAVE CSV
    # ========================================================

    importance_df.to_csv(
        CSV_PATH,
        index=False
    )

    # ========================================================
    # SAVE GRAPH
    # ========================================================

    plt.figure(figsize=(10, 6))

    plt.barh(
        importance_df["Feature"],
        importance_df["Mean_Absolute_SHAP_Value"],
        color="steelblue"
    )

    plt.xlabel("Mean Absolute SHAP Value")
    plt.ylabel("SCADA Feature")

    plt.title(
        "BO-BiGRU: Wind Turbine Anomaly Feature Importance"
    )

    plt.gca().invert_yaxis()

    plt.tight_layout()

    plt.savefig(
        IMAGE_PATH,
        dpi=300
    )

    plt.close()

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print("\nCSV SAVED  :", CSV_PATH)
    print("CHART SAVED:", IMAGE_PATH)

    print(
        "\nBO-BiGRU SHAP ANALYSIS COMPLETED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()