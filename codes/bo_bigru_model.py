import numpy as np
import tensorflow as tf
import optuna

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Bidirectional, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)


# ============================================================
# 1. LOAD PREPROCESSED DATA
# ============================================================

X_train = np.load("datasets/X_train_seq.npy")
y_train = np.load("datasets/y_train_seq.npy")

X_test = np.load("datasets/X_test_seq.npy")
y_test = np.load("datasets/y_test_seq.npy")


print("\n========================================")
print(" DATA LOADED")
print("========================================")

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_test :", X_test.shape)
print("y_test :", y_test.shape)


# ============================================================
# 2. CHECK CLASS DISTRIBUTION
# ============================================================

print("\n========================================")
print(" CLASS DISTRIBUTION")
print("========================================")

print("Training classes:")

unique_train, counts_train = np.unique(
    y_train,
    return_counts=True
)

for cls, count in zip(unique_train, counts_train):
    print(f"Class {cls}: {count}")


print("\nTesting classes:")

unique_test, counts_test = np.unique(
    y_test,
    return_counts=True
)

for cls, count in zip(unique_test, counts_test):
    print(f"Class {cls}: {count}")


# ============================================================
# 3. CREATE STRATIFIED VALIDATION SET
# ============================================================
#
# The original BO version produced val_auc = 0.0000.
# Therefore, we create a validation set containing both
# normal and anomaly classes using stratification.
#
# The test set remains completely untouched.
# ============================================================

X_train_bo, X_val_bo, y_train_bo, y_val_bo = train_test_split(

    X_train,
    y_train,

    test_size=0.20,

    random_state=42,

    stratify=y_train
)


print("\n========================================")
print(" BO VALIDATION DATA")
print("========================================")

print("BO Train:", X_train_bo.shape)
print("BO Validation:", X_val_bo.shape)

print("\nValidation class distribution:")

unique_val, counts_val = np.unique(
    y_val_bo,
    return_counts=True
)

for cls, count in zip(unique_val, counts_val):
    print(f"Class {cls}: {count}")


# ============================================================
# 4. CLASS WEIGHTS
# ============================================================

classes = np.unique(y_train_bo)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train_bo
)

class_weights = dict(
    zip(classes, weights)
)


print("\n========================================")
print(" CLASS WEIGHTS")
print("========================================")

print(class_weights)


# ============================================================
# 5. BAYESIAN OPTIMIZATION OBJECTIVE
# ============================================================

def objective(trial):

    # --------------------------------------------------------
    # Hyperparameters selected by Optuna
    # --------------------------------------------------------

    gru_units_1 = trial.suggest_categorical(
        "gru_units_1",
        [32, 64, 96]
    )

    gru_units_2 = trial.suggest_categorical(
        "gru_units_2",
        [16, 32, 48]
    )

    dropout_rate = trial.suggest_float(
        "dropout_rate",
        0.2,
        0.5,
        step=0.1
    )

    dense_units = trial.suggest_categorical(
        "dense_units",
        [16, 32, 64]
    )

    learning_rate = trial.suggest_float(
        "learning_rate",
        1e-4,
        1e-2,
        log=True
    )

    batch_size = trial.suggest_categorical(
        "batch_size",
        [128, 256, 512]
    )


    # --------------------------------------------------------
    # BUILD BiGRU
    # --------------------------------------------------------

    model = Sequential([

        Input(
            shape=(
                X_train.shape[1],
                X_train.shape[2]
            )
        ),

        Bidirectional(
            GRU(
                gru_units_1,
                return_sequences=True
            )
        ),

        Dropout(dropout_rate),

        Bidirectional(
            GRU(gru_units_2)
        ),

        Dropout(dropout_rate),

        Dense(
            dense_units,
            activation="relu"
        ),

        Dropout(0.2),

        Dense(
            1,
            activation="sigmoid"
        )
    ])


    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate
    )


    # --------------------------------------------------------
    # COMPILE
    # --------------------------------------------------------

    model.compile(

        optimizer=optimizer,

        loss="binary_crossentropy",

        metrics=[

            "accuracy",

            tf.keras.metrics.Precision(
                name="precision"
            ),

            tf.keras.metrics.Recall(
                name="recall"
            ),

            tf.keras.metrics.AUC(
                name="auc"
            )
        ]
    )


    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    early_stopping = EarlyStopping(

        monitor="val_auc",

        mode="max",

        patience=2,

        restore_best_weights=True
    )


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    history = model.fit(

        X_train_bo,

        y_train_bo,

        epochs=5,

        batch_size=batch_size,

        validation_data=(
            X_val_bo,
            y_val_bo
        ),

        class_weight=class_weights,

        callbacks=[early_stopping],

        verbose=0
    )


    # --------------------------------------------------------
    # BEST VALIDATION AUC
    # --------------------------------------------------------

    best_val_auc = max(
        history.history["val_auc"]
    )


    print(
        f"\nTrial {trial.number + 1}"
    )

    print(
        "Validation AUC:",
        round(best_val_auc, 4)
    )


    print(
        "Parameters:",
        trial.params
    )


    # Clear TensorFlow memory
    tf.keras.backend.clear_session()


    return best_val_auc


# ============================================================
# 6. START BAYESIAN OPTIMIZATION
# ============================================================

print("\n========================================")
print(" STARTING BO-BiGRU OPTIMIZATION")
print("========================================")


study = optuna.create_study(

    direction="maximize",

    sampler=optuna.samplers.TPESampler(
        seed=42
    )
)


# 5 trials for initial optimization
study.optimize(

    objective,

    n_trials=5
)


# ============================================================
# 7. BEST PARAMETERS
# ============================================================

print("\n========================================")
print(" BEST BO PARAMETERS")
print("========================================")

print(
    "Best Validation AUC:",
    study.best_value
)


for parameter, value in study.best_params.items():

    print(
        parameter,
        ":",
        value
    )


# ============================================================
# 8. GET BEST PARAMETERS
# ============================================================

params = study.best_params


# ============================================================
# 9. BUILD FINAL OPTIMIZED BiGRU
# ============================================================

final_model = Sequential([

    Input(
        shape=(
            X_train.shape[1],
            X_train.shape[2]
        )
    ),

    Bidirectional(
        GRU(
            params["gru_units_1"],
            return_sequences=True
        )
    ),

    Dropout(
        params["dropout_rate"]
    ),

    Bidirectional(
        GRU(
            params["gru_units_2"]
        )
    ),

    Dropout(
        params["dropout_rate"]
    ),

    Dense(
        params["dense_units"],
        activation="relu"
    ),

    Dropout(0.2),

    Dense(
        1,
        activation="sigmoid"
    )
])


# ============================================================
# 10. COMPILE FINAL MODEL
# ============================================================

final_optimizer = tf.keras.optimizers.Adam(

    learning_rate=params["learning_rate"]
)


final_model.compile(

    optimizer=final_optimizer,

    loss="binary_crossentropy",

    metrics=[

        "accuracy",

        tf.keras.metrics.Precision(
            name="precision"
        ),

        tf.keras.metrics.Recall(
            name="recall"
        ),

        tf.keras.metrics.AUC(
            name="auc"
        )
    ]
)


# ============================================================
# 11. FINAL MODEL SUMMARY
# ============================================================

print("\n========================================")
print(" OPTIMIZED BiGRU MODEL")
print("========================================")

final_model.summary()


# ============================================================
# 12. FINAL TRAINING
# ============================================================

print("\n========================================")
print(" TRAINING OPTIMIZED BO-BiGRU")
print("========================================")


final_early_stopping = EarlyStopping(

    monitor="val_auc",

    mode="max",

    patience=3,

    restore_best_weights=True
)


final_model.fit(

    X_train_bo,

    y_train_bo,

    epochs=10,

    batch_size=params["batch_size"],

    validation_data=(
        X_val_bo,
        y_val_bo
    ),

    class_weight=class_weights,

    callbacks=[
        final_early_stopping
    ],

    verbose=1
)


# ============================================================
# 13. TEST PREDICTION
# ============================================================

print("\n========================================")
print(" TESTING BO-BiGRU")
print("========================================")


y_probability = final_model.predict(

    X_test,

    batch_size=params["batch_size"]
).ravel()


# ============================================================
# 14. CLASSIFICATION
# ============================================================

threshold = 0.5

y_pred = (

    y_probability >= threshold

).astype(int)


# ============================================================
# 15. ACCURACY
# ============================================================

accuracy = accuracy_score(

    y_test,

    y_pred
)


print("\n========================================")
print(" BO-BiGRU RESULTS")
print("========================================")


print(
    "Accuracy:",
    round(accuracy, 4)
)


# ============================================================
# 16. CLASSIFICATION REPORT
# ============================================================

print("\n========================================")
print(" CLASSIFICATION REPORT")
print("========================================")


print(

    classification_report(

        y_test,

        y_pred,

        digits=4

    )

)


# ============================================================
# 17. CONFUSION MATRIX
# ============================================================

print("\n========================================")
print(" CONFUSION MATRIX")
print("========================================")


cm = confusion_matrix(

    y_test,

    y_pred

)


print(cm)


# ============================================================
# 18. SAVE OPTIMIZED MODEL
# ============================================================

final_model.save(

    "datasets/bo_bigru_anomaly_model.keras"

)


print("\n========================================")
print(" MODEL SAVED")
print("========================================")


print(
    "datasets/bo_bigru_anomaly_model.keras"
)


# ============================================================
# 19. FINAL MESSAGE
# ============================================================

print("\n========================================")
print(" BO-BiGRU COMPLETED SUCCESSFULLY")
print("========================================")