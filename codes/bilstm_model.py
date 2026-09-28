import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# ==========================================
# 1. LOAD PREPROCESSED DATA
# ==========================================

X_train = np.load("datasets/X_train_seq.npy")
y_train = np.load("datasets/y_train_seq.npy")

X_test = np.load("datasets/X_test_seq.npy")
y_test = np.load("datasets/y_test_seq.npy")

print("===== DATA LOADED =====")

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# ==========================================
# 2. CLASS WEIGHTS
# ==========================================

classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = dict(zip(classes, weights))

print("\n===== CLASS WEIGHTS =====")
print(class_weights)


# ==========================================
# 3. BUILD BiLSTM MODEL
# ==========================================

model = Sequential([

    Input(shape=(X_train.shape[1], X_train.shape[2])),

    Bidirectional(
        LSTM(64, return_sequences=True)
    ),

    Dropout(0.3),

    Bidirectional(
        LSTM(32)
    ),

    Dropout(0.3),

    Dense(32, activation="relu"),

    Dropout(0.2),

    Dense(1, activation="sigmoid")
])


# ==========================================
# 4. COMPILE MODEL
# ==========================================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)


print("\n===== MODEL SUMMARY =====")
model.summary()


# ==========================================
# 5. EARLY STOPPING
# ==========================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)


# ==========================================
# 6. TRAIN MODEL
# ==========================================

print("\n===== TRAINING BiLSTM =====")

history = model.fit(

    X_train,
    y_train,

    epochs=10,

    batch_size=256,

    validation_split=0.1,

    shuffle=False,

    class_weight=class_weights,

    callbacks=[early_stopping],

    verbose=1
)


# ==========================================
# 7. TEST PREDICTION
# ==========================================

print("\n===== TESTING =====")

y_probability = model.predict(
    X_test,
    batch_size=256
).ravel()

# Default threshold
threshold = 0.5

y_pred = (
    y_probability >= threshold
).astype(int)


# ==========================================
# 8. EVALUATION
# ==========================================

print("\n===== RESULTS =====")

print(
    "Accuracy:",
    accuracy_score(y_test, y_pred)
)

print("\n===== CLASSIFICATION REPORT =====")

print(
    classification_report(
        y_test,
        y_pred,
        digits=4
    )
)


# ==========================================
# 9. CONFUSION MATRIX
# ==========================================

print("\n===== CONFUSION MATRIX =====")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ==========================================
# 10. SAVE MODEL
# ==========================================

model.save(
    "datasets/bilstm_anomaly_model.keras"
)

print("\n===== MODEL SAVED =====")
print("bilstm_anomaly_model.keras")