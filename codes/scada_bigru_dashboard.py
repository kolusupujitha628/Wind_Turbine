from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="BO-BiGRU Wind Turbine SCADA Monitor",
    page_icon="⚙️",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "datasets"
RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_PATH = DATASET_DIR / "bo_bigru_anomaly_model.keras"
X_TEST_PATH = DATASET_DIR / "X_test_seq.npy"

SHAP_PATH = RESULTS_DIR / "bo_bigru_shap_feature_importance.csv"


# ============================================================
# SCADA FEATURES
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
# LOAD MODEL AND DATA
# ============================================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        MODEL_PATH
    )


@st.cache_data
def load_data():

    X_test = np.load(
        X_TEST_PATH
    ).astype("float32")

    shap_df = pd.read_csv(
        SHAP_PATH
    )

    return X_test, shap_df


# ============================================================
# HEALTH STATUS
# ============================================================

def get_status(health_score):

    if health_score >= 80:
        return "HEALTHY", "success"

    if health_score >= 60:
        return "WATCH", "info"

    if health_score >= 40:
        return "WARNING", "warning"

    return "CRITICAL", "error"


# ============================================================
# MAINTENANCE RECOMMENDATION
# ============================================================

def get_recommendation(
    status,
    features
):

    feature_text = ", ".join(features)

    if status == "HEALTHY":

        return (
            "Turbine is healthy. Continue normal "
            "SCADA monitoring."
        )

    if status == "WATCH":

        return (
            f"Monitor these parameters carefully: "
            f"{feature_text}. Plan inspection in the "
            "next maintenance cycle."
        )

    if status == "WARNING":

        return (
            f"Possible gearbox or bearing degradation "
            f"detected. Check: {feature_text}. "
            "Schedule maintenance soon."
        )

    return (
        f"Critical fault risk detected. Inspect immediately: "
        f"{feature_text}. Consider stopping the turbine "
        "to avoid severe damage."
    )


# ============================================================
# DASHBOARD HEADER
# ============================================================

st.title(
    "⚙️ BO-BiGRU Wind Turbine SCADA Health Monitoring Dashboard"
)

st.caption(
    "BO-BiGRU • SCADA Data • SHAP Explainability • "
    "Dynamic Health Score"
)


# ============================================================
# VALIDATION
# ============================================================

if not MODEL_PATH.exists():

    st.error("BO-BiGRU model file not found.")

    st.stop()

if not SHAP_PATH.exists():

    st.error(
        "BO-BiGRU SHAP result not found. "
        "Run shap_bigru_analysis.py first."
    )

    st.stop()


# ============================================================
# LOAD RESOURCES
# ============================================================

model = load_model()

X_test, shap_df = load_data()


# ============================================================
# SIDEBAR CONTROL
# ============================================================

st.sidebar.header("SCADA Control Panel")

sequence_index = st.sidebar.slider(

    "Select SCADA sequence",

    min_value=0,

    max_value=len(X_test) - 1,

    value=len(X_test) - 1
)

sequence = X_test[sequence_index]


# ============================================================
# BO-BiGRU PREDICTION
# ============================================================

anomaly_probability = float(

    model.predict(

        np.expand_dims(sequence, axis=0),

        verbose=0

    ).reshape(-1)[0]
)

health_score = (
    1 - anomaly_probability
) * 100

status, status_type = get_status(
    health_score
)

top_features = shap_df[
    "Feature"
].head(3).tolist()

recommendation = get_recommendation(
    status,
    top_features
)


# ============================================================
# HEALTH METRICS
# ============================================================

col1, col2, col3 = st.columns(3)

col1.metric(
    "Health Score",
    f"{health_score:.2f} / 100"
)

col2.metric(
    "Anomaly Probability",
    f"{anomaly_probability:.4f}"
)

col3.metric(
    "Turbine Status",
    status
)


# ============================================================
# STATUS ALERT
# ============================================================

if status_type == "success":

    st.success(recommendation)

elif status_type == "info":

    st.info(recommendation)

elif status_type == "warning":

    st.warning(recommendation)

else:

    st.error(recommendation)


# ============================================================
# SCADA SENSOR TREND
# ============================================================

st.subheader("SCADA Sensor Trend")

st.caption(
    "Current 20-step scaled SCADA sensor sequence "
    "used by the BO-BiGRU model."
)

sensor_df = pd.DataFrame(
    sequence,
    columns=FEATURE_NAMES
)

sensor_df.index.name = "Time Step"

st.line_chart(
    sensor_df
)


# ============================================================
# SHAP FEATURE IMPORTANCE
# ============================================================

st.subheader("BO-BiGRU SHAP Feature Importance")

st.caption(
    "Features with higher SHAP values have greater "
    "influence on anomaly prediction."
)

importance_df = shap_df.sort_values(

    by="Mean_Absolute_SHAP_Value",

    ascending=False
)

st.bar_chart(

    importance_df.set_index(
        "Feature"
    )["Mean_Absolute_SHAP_Value"]
)

st.dataframe(

    importance_df,

    width="stretch",

    hide_index=True
)


# ============================================================
# FUTURE HARDWARE EXTENSION
# ============================================================

st.subheader("Future Hardware Integration")

st.write(

    "Temperature + Vibration + Pressure sensors → "
    "ESP32/Raspberry Pi → Live SCADA values → "
    "BO-BiGRU Dashboard"
)

st.caption(
    "Current dashboard uses stored SCADA data "
    "in software demo mode."
)