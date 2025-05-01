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

# === Check and Drop Missing Values ===
print(f"Before cleaning: {df.isnull().sum().sum()} total missing values.")
df = df.dropna()
print(f"After dropping missing: {df.isnull().sum().sum()} total missing values.")

# === Identify Categorical Columns ===
categorical_cols = df.select_dtypes(include=['object']).columns

if len(categorical_cols) > 0:
    print(f"🎯 Encoding {len(categorical_cols)} categorical columns:")
    print(f"    ➤ {list(categorical_cols)}")

label_encoders = {}

# === Encode Each Categorical Column ===
for col in categorical_cols:
    print(f"🔧 Encoding column: {col}")
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str).fillna("Unknown"))
    label_encoders[col] = le

# === Save Encoded Data ===
os.makedirs(output_folder, exist_ok=True)
df.to_csv(output_file, index=False)
print(f"\n✅ Encoded dataset saved to: {output_file}")
print(f"📊 Rows: {len(df)} | Columns: {len(df.columns)}")
