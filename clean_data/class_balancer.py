# class_balance.py

import pandas as pd
import os

# === Configuration ===
input_file = r"Chicago_cleaned_output\Chicago_cleaned.csv"
output_folder = r"Chicago_cleaned_output"
balanced_file = os.path.join(output_folder, "Chicago_balanced.csv")

os.makedirs(output_folder, exist_ok=True)

# === Load Cleaned Data ===
print("📂 Loading cleaned dataset...")
df = pd.read_csv(input_file)

# === Separate Classes ===
violent_df = df[df['is_violent'] == 1]
non_violent_df = df[df['is_violent'] == 0]

print(f"🔍 Violent crimes: {len(violent_df)}")
print(f"🔍 Non-violent crimes: {len(non_violent_df)}")

# === Undersample Non-Violent Crimes ===
balanced_non_violent_df = non_violent_df.sample(n=len(violent_df), random_state=42)

# === Combine and Shuffle ===
balanced_df = pd.concat([violent_df, balanced_non_violent_df])
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# === Save Balanced Dataset ===
balanced_df.to_csv(balanced_file, index=False)
print(f"✅ Balanced dataset saved to: {balanced_file} ({len(balanced_df)} rows)")
