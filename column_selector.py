import pandas as pd
import os
import json

# === Configuration ===
input_file = r"D:\Data\Data Mining 2025\Chicago\Crimes_-_2001_to_Present_20250220.csv"
iucr_mapping_file = "iucr_codes.json"  # Ensure this is in the same directory as your script
output_folder = "Chicago_cleaned_output"
output_file = os.path.join(output_folder, "Chicago_cleaned.csv")

# === Ensure Output Directory Exists ===
os.makedirs(output_folder, exist_ok=True)

# === Load IUCR Mapping ===
with open(iucr_mapping_file, "r") as f:
    iucr_dict = json.load(f)

# === Load Dataset with Only Needed Columns ===
columns_to_keep = ["IUCR", "Date", "District", "Location Description", "Location"]
df = pd.read_csv(input_file, usecols=lambda col: col in columns_to_keep)

# Ensure IUCR is a zero-padded 4-character string
df['IUCR'] = df['IUCR'].astype(str).str.zfill(4)

# === Extract AM/PM from original Date string ===
df['AM_PM'] = df['Date'].str.extract(r'(AM|PM)')

# === Convert Date to Datetime and Split Components ===
df['DateTime'] = pd.to_datetime(df['Date'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
df['Date'] = df['DateTime'].dt.date
df['Time'] = df['DateTime'].dt.time
df['Hour'] = df['DateTime'].dt.hour
df['Month'] = df['DateTime'].dt.month

# === Extract Latitude and Longitude from Location column ===
df[['Latitude', 'Longitude']] = df['Location'].str.extract(r'\((.*), (.*)\)').astype(float)

# === Classify Crimes by IUCR Index Code ===
def is_violent(iucr):
    entry = iucr_dict.get(str(iucr).strip())
    if entry:
        return 1 if entry.get("Index") == "I" else 0
    return None

def get_primary(iucr):
    return iucr_dict.get(str(iucr).strip(), {}).get("Primary")

def get_secondary(iucr):
    return iucr_dict.get(str(iucr).strip(), {}).get("Secondary")

df['is_violent'] = df['IUCR'].apply(is_violent)
df['Crime_Type'] = df['IUCR'].apply(get_primary)
df['Crime_Subtype'] = df['IUCR'].apply(get_secondary)

# === Drop rows with missing critical values ===
df = df.dropna(subset=['IUCR', 'District', 'DateTime', 'Latitude', 'Longitude'])

# === Drop raw Location string and DateTime helper column ===
df = df.drop(columns=['Location', 'DateTime'])

# === Final column order ===
df = df[['IUCR', 'Crime_Type', 'Crime_Subtype', 'is_violent',
         'District', 'Latitude', 'Longitude', 'Location Description',
         'Date', 'Time', 'Hour', 'AM_PM', 'Month']]

# === Save Cleaned CSV ===
df.to_csv(output_file, index=False)
print(f"✅ Final cleaned dataset with violence classification saved to: {output_file} ({len(df)} rows, {len(df.columns)} columns)")
