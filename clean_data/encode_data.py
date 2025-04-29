# encode_dataset.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

# === Configuration ===
input_file = r"C:\Users\trend\PycharmProjects\data-group-2\Chicago_cleaned_output\Chicago_cleaned.csv"
output_folder = r"C:\Users\trend\PycharmProjects\data-group-2\Chicago_cleaned_output"
output_file = os.path.join(output_folder, "Chicago_encoded.csv")

# === Load Data ===
print("📥 Loading cleaned dataset...")
df = pd.read_csv(input_file, dtype={'IUCR': 'object'})

# === Check for Missing Values ===
print(f"Before cleaning: {df.isnull().sum().sum()} total missing values.")
df = df.dropna()
print(f"After dropping missing: {df.isnull().sum().sum()} total missing values.")

# === Label Encode Categorical Features ===
categorical_cols = df.select_dtypes(include=['object']).columns

if len(categorical_cols) > 0:
    print(f"🎯 Encoding {len(categorical_cols)} categorical columns: {list(categorical_cols)}")

label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# === Save Encoded Data ===
os.makedirs(output_folder, exist_ok=True)
df.to_csv(output_file, index=False)
print(f"✅ Encoded dataset saved to: {output_file} ({len(df)} rows, {len(df.columns)} columns)")
