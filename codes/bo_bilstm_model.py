"""Bayesian-optimised BiLSTM for wind-turbine SCADA anomaly classification."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
import keras_tuner as kt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# SETTINGS
# ============================================================

SEED = 42

# Laptop CPU ki practical settings
MAX_TRIALS = 5
EPOCHS_PER_TRIAL = 5
FINAL_EPOCHS = 10
BATCH_SIZE = 256


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "datasets"

MODEL_PATH = DATASET_DIR / "bo_bilstm_anomaly_model.keras"
TUNER_DIR = PROJECT_ROOT / "tuner_results"


# ============================================================
# SET RANDOM SEED
# ============================================================

def set_seed():

    os.environ["PYTHONHASHSEED"] = str(SEED)

    random.seed(SEED)
    np.random.seed(SEED)

    tf.keras.utils.set_random_seed(SEED)


# ============================================================
# LABEL CONVERSION
# ============================================================

def to_class_ids(y):

    y = np.asarray(y)

    # One-hot labels unte class IDs ki convert chestundi
    if y.ndim > 1 and y.shape[-1] > 1:
        return np.argmax(y, axis=-1).astype("int32")

    return y.reshape(-1).astype("int32")


# ============================================================
# BUILD BiLSTM MODEL
# ============================================================

def build_model(hp, input_shape, n_classes):

    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            units=hp.Int(
                "lstm_units",
                min_value=32,
                max_value=160,
                step=32
            ),

            return_sequences=False,

            dropout=hp.Float(
                "lstm_dropout",
                min_value=0.0,
                max_value=0.4,
                step=0.1
            ),

            recurrent_dropout=0.0
        ),

        name="bidirectional_lstm"
    )(inputs)

    x = tf.keras.layers.Dense(
        hp.Int(
            "dense_units",
            min_value=16,
            max_value=128,
            step=16
        ),

        activation="relu"
    )(x)

    x = tf.keras.layers.Dropout(
        hp.Float(
            "dense_dropout",
            min_value=0.0,
            max_value=0.5,
            step=0.1
        )
    )(x)

    # Binary anomaly classification
    if n_classes == 2:

        outputs = tf.keras.layers.Dense(
            1,
            activation="sigmoid",
            name="anomaly_output"
        )(x)

        loss = "binary_crossentropy"

    # Multiclass classification support
    else:

        outputs = tf.keras.layers.Dense(
            n_classes,
            activation="softmax",
            name="class_output"
        )(x)

        loss = "sparse_categorical_crossentropy"

    model = tf.keras.Model(
        inputs,
        outputs,
        name="BO_BiLSTM"
    )

    model.compile(

        optimizer=tf.keras.optimizers.Adam(

            learning_rate=hp.Choice(
                "learning_rate",
                [1e-2, 3e-3, 1e-3, 3e-4]
            )
        ),

        loss=loss,

        metrics=["accuracy"]
    )

    return model


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    set_seed()

    DATASET_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    required_files = [

        "X_train_seq.npy",
        "X_test_seq.npy",
        "y_train_seq.npy",
        "y_test_seq.npy"
    ]

    missing = [

        file_name for file_name in required_files
        if not (DATASET_DIR / file_name).exists()
    ]

    if missing:

        raise FileNotFoundError(
            "Missing dataset file(s): " + ", ".join(missing)
        )

    # ========================================================
    # LOAD DATA
    # ========================================================

    X_train = np.load(
        DATASET_DIR / "X_train_seq.npy"
    ).astype("float32")

    X_test = np.load(
        DATASET_DIR / "X_test_seq.npy"
    ).astype("float32")

    y_train = to_class_ids(
        np.load(DATASET_DIR / "y_train_seq.npy")
    )

    y_test = to_class_ids(
        np.load(DATASET_DIR / "y_test_seq.npy")
    )

    if X_train.ndim != 3 or X_test.ndim != 3:

        raise ValueError(

            "BiLSTM expects 3-D data: "
            "(samples, time_steps, features). "
            f"Received train shape {X_train.shape} "
            f"and test shape {X_test.shape}."
        )

    all_classes = np.unique(
        np.concatenate([y_train, y_test])
    )

    # Labels {1,2} laga unte {0,1} ga map chestundi
    if not np.array_equal(
        all_classes,
        np.arange(len(all_classes))
    ):

        class_map = {

            label: index
            for index, label in enumerate(all_classes)
        }

        y_train = np.array(
            [class_map[label] for label in y_train],
            dtype="int32"
        )

        y_test = np.array(
            [class_map[label] for label in y_test],
            dtype="int32"
        )

    n_classes = len(all_classes)

    if n_classes < 2:

        raise ValueError(
            "At least two classes are required."
        )

    print("\n========== BO-BiLSTM WIND TURBINE ANOMALY MODEL ==========")

    print("Training sequences :", X_train.shape)
    print("Testing sequences  :", X_test.shape)
    print("Number of classes  :", n_classes)

    # ========================================================
    # TRAIN / VALIDATION SPLIT
    # ========================================================

    X_fit, X_val, y_fit, y_val = train_test_split(

        X_train,
        y_train,

        test_size=0.20,

        random_state=SEED,

        stratify=y_train
    )

    # ========================================================
    # CLASS WEIGHTS
    # ========================================================

    weights = compute_class_weight(

        class_weight="balanced",

        classes=np.unique(y_fit),

        y=y_fit
    )

    class_weight = dict(

        zip(
            np.unique(y_fit),
            weights
        )
    )

    print("\nClass weights:")

    print(class_weight)

    # ========================================================
    # BAYESIAN OPTIMIZATION
    # ========================================================

    tuner = kt.BayesianOptimization(

        hypermodel=lambda hp: build_model(
            hp,
            X_train.shape[1:],
            n_classes
        ),

        objective=kt.Objective(
            "val_accuracy",
            direction="max"
        ),

        max_trials=MAX_TRIALS,

        num_initial_points=5,

        directory=str(TUNER_DIR),

        # New folder name: old OneDrive-locked folder problem avoid chestundi
        project_name="bo_bilstm_run2",

        overwrite=True,

        seed=SEED
    )

    callbacks = [

        tf.keras.callbacks.EarlyStopping(

            monitor="val_loss",

            patience=3,

            restore_best_weights=True
        )
    ]

    print("\n========== STARTING BAYESIAN OPTIMIZATION ==========")

    tuner.search(

        X_fit,
        y_fit,

        validation_data=(X_val, y_val),

        epochs=EPOCHS_PER_TRIAL,

        batch_size=BATCH_SIZE,

        class_weight=class_weight,

        callbacks=callbacks,

        verbose=1
    )

    # ========================================================
    # BEST HYPERPARAMETERS
    # ========================================================

    best_hp = tuner.get_best_hyperparameters(1)[0]

    print("\n========== BEST HYPERPARAMETERS ==========")

    for name, value in best_hp.values.items():

        print(f"{name}: {value}")

    # ========================================================
    # FINAL MODEL TRAINING
    # ========================================================

    best_model = build_model(

        best_hp,

        X_train.shape[1:],

        n_classes
    )

    final_callbacks = [

        tf.keras.callbacks.EarlyStopping(

            monitor="val_loss",

            patience=5,

            restore_best_weights=True
        )
    ]

    print("\n========== TRAINING FINAL BO-BiLSTM MODEL ==========")

    best_model.fit(

        X_fit,
        y_fit,

        validation_data=(X_val, y_val),

        epochs=FINAL_EPOCHS,

        batch_size=BATCH_SIZE,

        class_weight=class_weight,

        callbacks=final_callbacks,

        verbose=1
    )

    # ========================================================
    # TEST PREDICTION
    # ========================================================

    probabilities = best_model.predict(

        X_test,

        batch_size=BATCH_SIZE,

        verbose=0
    )

    if n_classes == 2:

        y_pred = (

            probabilities.reshape(-1) >= 0.5

        ).astype("int32")

    else:

        y_pred = np.argmax(
            probabilities,
            axis=1
        )

    # ========================================================
    # RESULTS
    # ========================================================

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    f1 = f1_score(

        y_test,
        y_pred,

        average="weighted",

        zero_division=0
    )

    print("\n==================== TEST RESULTS ====================")

    print(f"Test accuracy     : {accuracy:.4f}")
    print(f"Weighted F1-score : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )

    print("\nClassification Report:")

    print(

        classification_report(

            y_test,
            y_pred,

            digits=4,

            zero_division=0
        )
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    best_model.save(
        MODEL_PATH
    )

    print(f"\nMODEL SAVED: {MODEL_PATH}")

    print("BO-BiLSTM COMPLETED SUCCESSFULLY")


if __name__ == "__main__":

    main()