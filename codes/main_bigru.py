import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "datasets"
RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_PATH = DATASET_DIR / "bo_bigru_anomaly_model.keras"
X_TEST_PATH = DATASET_DIR / "X_test_seq.npy"

SHAP_PATH = RESULTS_DIR / "bo_bigru_shap_feature_importance.csv"

JSON_REPORT_PATH = RESULTS_DIR / "bo_bigru_latest_health_report.json"
TEXT_REPORT_PATH = RESULTS_DIR / "bo_bigru_latest_health_report.txt"

RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# HEALTH STATUS
# ============================================================

def get_health_status(health_score):

    if health_score >= 80:
        return "HEALTHY"

    if health_score >= 60:
        return "WATCH"

    if health_score >= 40:
        return "WARNING"

    return "CRITICAL"


# ============================================================
# MAINTENANCE RECOMMENDATION
# ============================================================

def get_maintenance_recommendation(
    status,
    important_features
):

    feature_text = ", ".join(important_features)

    if status == "HEALTHY":

        return (
            "Turbine condition is healthy. Continue normal "
            "SCADA monitoring and scheduled maintenance."
        )

    if status == "WATCH":

        return (
            f"Monitor the turbine closely. Important features: "
            f"{feature_text}. Plan inspection during the next "
            "maintenance window."
        )

    if status == "WARNING":

        return (
            f"Possible gearbox or bearing degradation detected. "
            f"Inspect: {feature_text}. "
            "Schedule maintenance soon."
        )

    return (
        f"Critical anomaly risk detected. Important features: "
        f"{feature_text}. Perform immediate inspection and "
        "consider stopping the turbine to prevent severe damage."
    )


# ============================================================
# PREDICT HEALTH FROM ONE SCADA SEQUENCE
# ============================================================

def predict_turbine_health(
    model,
    sequence,
    important_features
):

    sequence = np.asarray(
        sequence,
        dtype="float32"
    )

    if sequence.shape != (20, 7):

        raise ValueError(

            "Expected one SCADA sequence with shape (20, 7). "

            f"Received: {sequence.shape}"
        )

    probability = model.predict(

        np.expand_dims(sequence, axis=0),

        verbose=0

    ).reshape(-1)[0]

    anomaly_probability = float(probability)

    health_score = float(
        (1 - anomaly_probability) * 100
    )

    status = get_health_status(
        health_score
    )

    recommendation = get_maintenance_recommendation(
        status,
        important_features
    )

    return {

        "anomaly_probability": round(
            anomaly_probability,
            4
        ),

        "health_score": round(
            health_score,
            2
        ),

        "health_status": status,

        "maintenance_recommendation": recommendation
    }


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print("\n========== BO-BiGRU WIND TURBINE HEALTH MONITOR ==========")

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"BO-BiGRU model not found: {MODEL_PATH}"
        )

    if not SHAP_PATH.exists():

        raise FileNotFoundError(
            "BO-BiGRU SHAP file not found. "
            "Run shap_bigru_analysis.py first."
        )

    # Load BO-BiGRU model
    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    # Latest available SCADA sequence
    X_test = np.load(
        X_TEST_PATH
    ).astype("float32")

    latest_sequence = X_test[-1]

    # Load BO-BiGRU SHAP top features
    shap_df = pd.read_csv(
        SHAP_PATH
    )

    important_features = shap_df[
        "Feature"
    ].head(3).tolist()

    # Predict turbine health
    health_result = predict_turbine_health(

        model,

        latest_sequence,

        important_features
    )

    report = {

        "report_time": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),

        "model": "BO-BiGRU",

        "mode": "Software Demo using latest SCADA sequence",

        "top_shap_features": important_features,

        **health_result
    }

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print("\n========== CURRENT TURBINE HEALTH ==========")

    print(
        "Anomaly Probability :",
        report["anomaly_probability"]
    )

    print(
        "Health Score        :",
        report["health_score"],
        "/ 100"
    )

    print(
        "Health Status       :",
        report["health_status"]
    )

    print("\nTop Important Features:")

    for feature in important_features:

        print("-", feature)

    print("\nMaintenance Recommendation:")

    print(
        report["maintenance_recommendation"]
    )

    # ========================================================
    # SAVE JSON REPORT
    # ========================================================

    with open(
        JSON_REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )

    # ========================================================
    # SAVE TEXT REPORT
    # ========================================================

    with open(
        TEXT_REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "BO-BiGRU WIND TURBINE HEALTH REPORT\n"
        )

        file.write(
            "=" * 40 + "\n"
        )

        file.write(
            f"Report Time: {report['report_time']}\n"
        )

        file.write(
            f"Model: {report['model']}\n"
        )

        file.write(
            f"Health Score: {report['health_score']} / 100\n"
        )

        file.write(
            f"Health Status: {report['health_status']}\n"
        )

        file.write(
            "Top Features: "
            + ", ".join(important_features)
            + "\n"
        )

        file.write(
            "Recommendation: "
            + report["maintenance_recommendation"]
            + "\n"
        )

    print("\nJSON REPORT SAVED :", JSON_REPORT_PATH)

    print("TEXT REPORT SAVED :", TEXT_REPORT_PATH)

    print(
        "\nBO-BiGRU DYNAMIC HEALTH MONITOR "
        "COMPLETED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()