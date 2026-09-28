import pandas as pd

# Load labeled dataset
df = pd.read_csv("datasets/turbine_5yr_labeled_data.csv")

print("\n===== DATASET SHAPE =====")
print(df.shape)

print("\n===== COLUMN NAMES =====")
print(df.columns.tolist())

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== ANOMALY DISTRIBUTION =====")
print(df["is_anomaly"].value_counts())

print("\n===== DATA TYPES =====")
print(df.dtypes)