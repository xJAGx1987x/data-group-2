import pandas as pd
import os
from sklearn.model_selection import train_test_split

# === Configuration ===
input_file = r"Chicago_cleaned_output\Chicago_cleaned.csv"
output_folder = "Chicago_cleaned_output"

train_output = os.path.join(output_folder, "Chicago_train.csv")
test_output = os.path.join(output_folder, "Chicago_test.csv")

# === Ensure Output Directory Exists ===
os.makedirs(output_folder, exist_ok=True)

# === Load the Full Cleaned CSV ===
df = pd.read_csv(input_file)

# === Train/Test Split ===
# You can control the size here: test_size=0.2 means 20% test set
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

# === Save the Splits ===
train_df.to_csv(train_output, index=False)
test_df.to_csv(test_output, index=False)

# === Confirm ===
print(f"✅ Training set saved: {train_output} ({train_df.shape[0]} rows)")
print(f"✅ Testing set saved: {test_output} ({test_df.shape[0]} rows)")
