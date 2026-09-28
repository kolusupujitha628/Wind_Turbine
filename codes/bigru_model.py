import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# ============================================================
# LOAD PREPROCESSED SEQUENCE DATA
# ============================================================

from preprocessing import (
    X_train_seq,
    y_train_seq,
    X_test_seq,
    y_test_seq
)

print("\n===== DATA SHAPES =====")
print("X_train:", X_train_seq.shape)
print("y_train:", y_train_seq.shape)
print("X_test :", X_test_seq.shape)
print("y_test :", y_test_seq.shape)


# ============================================================
# BUILD BiGRU MODEL
# ============================================================

model = Sequential([
    Bidirectional(
        GRU(64, return_sequences=True),
        input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])
    ),

    Dropout(0.3),

    Bidirectional(
        GRU(32)
    ),

    Dropout(0.3),

    Dense(32, activation="relu"),

    Dropout(0.2),

    Dense(1, activation="sigmoid")
])


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)


# ============================================================
# MODEL SUMMARY
# ============================================================

print("\n===== BiGRU MODEL =====")
model.summary()


# ============================================================
# TRAIN
# ============================================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

print("\n===== TRAINING BiGRU =====")

history = model.fit(
    X_train_seq,
    y_train_seq,
    epochs=10,
    batch_size=256,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)


# ============================================================
# TESTING
# ============================================================

print("\n===== TESTING =====")

y_probability = model.predict(
    X_test_seq,
    batch_size=256
)

y_pred = (y_probability >= 0.5).astype(int).ravel()


# ============================================================
# RESULTS
# ============================================================

accuracy = accuracy_score(
    y_test_seq,
    y_pred
)

print("\n===== RESULTS =====")
print("Accuracy:", accuracy)


print("\n===== CLASSIFICATION REPORT =====")

print(
    classification_report(
        y_test_seq,
        y_pred,
        digits=4
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n===== CONFUSION MATRIX =====")

cm = confusion_matrix(
    y_test_seq,
    y_pred
)

print(cm)


# ============================================================
# SAVE MODEL
# ============================================================

model.save("bigru_anomaly_model.keras")

print("\n===== MODEL SAVED =====")
print("bigru_anomaly_model.keras")