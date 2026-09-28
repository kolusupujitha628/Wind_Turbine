import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# ==============================
# 1. LOAD DATASET
# ==============================

df = pd.read_csv("datasets/turbine_5yr_labeled_data.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# Remove missing values
df = df.dropna().reset_index(drop=True)

print("===== DATASET =====")
print("Shape:", df.shape)


# ==============================
# 2. SELECT FEATURES
# ==============================

features = [
    "gearbox_oil_temp",
    "gearbox_bearing_temp",
    "vibration_x",
    "vibration_y",
    "vibration_z",
    "oil_pressure",
    "particle_count"
]

X = df[features]
y = df["is_anomaly"].values


# ==============================
# 3. CHRONOLOGICAL SPLIT
# ==============================

split_index = int(len(X) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]


# ==============================
# 4. FEATURE SCALING
# ==============================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ==============================
# 5. CREATE TIME SEQUENCES
# ==============================

sequence_length = 20


def create_sequences(X, y, sequence_length):

    X_seq = []
    y_seq = []

    for i in range(len(X) - sequence_length):

        X_seq.append(
            X[i:i + sequence_length]
        )

        # Label of the last timestep
        y_seq.append(
            y[i + sequence_length]
        )

    return np.array(X_seq), np.array(y_seq)


X_train_seq, y_train_seq = create_sequences(
    X_train_scaled,
    y_train,
    sequence_length
)

X_test_seq, y_test_seq = create_sequences(
    X_test_scaled,
    y_test,
    sequence_length
)


# ==============================
# 6. DISPLAY RESULTS
# ==============================

print("\n===== SEQUENCE DATA =====")

print("Sequence length:", sequence_length)

print("X_train sequence shape:",
      X_train_seq.shape)

print("y_train sequence shape:",
      y_train_seq.shape)

print("X_test sequence shape:",
      X_test_seq.shape)

print("y_test sequence shape:",
      y_test_seq.shape)


print("\n===== ANOMALY DISTRIBUTION =====")

print("Training:")
print(pd.Series(y_train_seq).value_counts())

print("\nTesting:")
print(pd.Series(y_test_seq).value_counts())


print("\n===== PREPROCESSING COMPLETED =====")
# Save preprocessed sequences
np.save("datasets/X_train_seq.npy", X_train_seq)
np.save("datasets/y_train_seq.npy", y_train_seq)
np.save("datasets/X_test_seq.npy", X_test_seq)
np.save("datasets/y_test_seq.npy", y_test_seq)

print("\n===== DATA SAVED =====")
print("Preprocessed files saved successfully.")