import pandas as pd
import os

# === Configuration ===
input_file = "Chicago_cleaned_output/Chicago_filtered_data.csv"
output_folder = "Chicago_cleaned_output"
sample_output = os.path.join(output_folder, "Chicago_sample_100k.csv")

# === Ensure Output Directory Exists ===
os.makedirs(output_folder, exist_ok=True)

# === Load the Full Cleaned CSV ===
df = pd.read_csv(input_file)

# === Sample 100,000 Rows ===
df_sample = df.sample(n=100_000, random_state=42)

# === Save Sample ===
df_sample.to_csv(sample_output, index=False)
print(f"📊 Sample saved for Excel: {sample_output}")
